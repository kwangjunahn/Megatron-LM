# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import gc
import os
import sys

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import get_mxfp8_block_scaling_recipe
from megatron.core.fp8_utils import is_float8tensor, is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import dequantize_gtp_native_fp8
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import force_param_sync, get_model, setup_model_and_optimizer
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()


def _get_mxfp8_2d_availability():
    try:
        get_mxfp8_block_scaling_recipe(mxfp8_2d_quantization=True)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return False, f"MXFP8 2D quantization is not available: {exc}"
    return True, ""


mxfp8_2d_available, reason_for_no_mxfp8_2d = _get_mxfp8_2d_availability()

cuda_graph_supported = False
reason_for_no_cuda_graph = ""
try:
    from transformer_engine.pytorch.tensor.utils import post_all_gather_processing

    if is_te_min_version("2.10.0"):
        cuda_graph_supported = True
    else:
        reason_for_no_cuda_graph = "Need newer TransformerEngine"
except ImportError:
    reason_for_no_cuda_graph = "Need newer TransformerEngine"


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    return (
        not args.use_megatron_fsdp and args.use_distributed_optimizer and args.overlap_param_gather
    )


class TestFP8Param:

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.cuda_graph_helper = None
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    def create_test_args(
        self,
        tp,
        recipe,
        sequence_length,
        micro_batch_size,
        inference,
        fp8_param_gather,
        use_cuda_graph,
        **kwargs,
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_fp8_param.py']
        args = parse_args()
        args.num_layers = 4
        args.padded_vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = not inference
        args.fp8 = "e4m3"
        args.fp8_recipe = recipe
        args.fp8_param_gather = fp8_param_gather
        args.ddp_bucket_size = 1024  # Create more buckets to test the rs/ag overlap.

        # MXFP8 test settings
        if recipe == "mxfp8" and fp8_param_gather:
            args.reuse_grad_buf_for_mxfp8_param_ag = True

        if use_cuda_graph:
            args.cuda_graph_impl = "transformer_engine"
            args.cuda_graph_warmup_steps = 0

        for key, value in kwargs.items():
            assert hasattr(args, key)
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def copy_main_params_to_param_buffer(self, model_chunks, optimizer):
        # Mirrors MBridge's pre-eval fix: disable_forward_pre_hook(param_sync=True)
        # force-syncs params before eval callbacks run, so MXFP8 must repopulate
        # the shared param/grad buffer before disabling forward hooks.
        for model_chunk in model_chunks:
            model_chunk.zero_grad_buffer()
        for optim_instance in optimizer.chained_optimizers:
            if isinstance(optim_instance, DistributedOptimizer):
                optim_instance._copy_main_params_to_param_buffer()

    def run_eval_transition(self, args, model_chunks, optimizer, batch):
        input_ids, labels, position_ids, attention_mask, loss_mask = batch

        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            self.copy_main_params_to_param_buffer(model_chunks, optimizer)

        if should_disable_forward_pre_hook(args):
            disable_forward_pre_hook(model_chunks, param_sync=True)

        model_chunks[0].eval()
        model_chunks[0].set_is_first_microbatch()
        with torch.no_grad():
            eval_output = model_chunks[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
        eval_loss = eval_output.mean()
        model_chunks[0].train()

        if should_disable_forward_pre_hook(args):
            enable_forward_pre_hook(model_chunks)

        return eval_loss.item()

    def _run_test_helper(
        self,
        tp_size,
        recipe,
        inference: bool = False,
        fp8_param_gather: bool = True,
        use_cuda_graph: bool = False,
        eval_transition: bool = False,
        **kwargs,
    ):
        """Test fp8_param with a small GPT model."""
        # Test-only knob: not a model arg, so pop before create_test_args (which asserts every
        # kwarg is a real arg attribute).
        save_at_steps_kw = kwargs.pop("save_at_steps", ())
        args = self.create_test_args(
            tp_size,
            recipe,
            self.seq_length,
            self.micro_batch_size,
            inference,
            fp8_param_gather,
            use_cuda_graph,
            **kwargs,
        )

        if recipe == "blockwise" and args.sequence_parallel:
            assert (
                tp_size * 128 <= self.seq_length
            ), "Blockwise recipe and sequence parallelism requires tp_size * 128 <= seq_length"

        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            # Enable GTP weight-remat when the test requested it (default 1 => no GTP, so
            # non-GTP fp8 tests are unaffected).
            gtp_remat_size=getattr(args, "gtp_weight_remat_size", 1),
        )

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )
        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        if inference:
            model_cfg = cfg_container.model
            builder_cls = model_cfg.get_builder_cls()
            builder = builder_cls(model_cfg)
            gpt_model = builder.build_distributed_models(
                pg_collection=pg_collection, wrap_with_ddp=False
            )
            gpt_model[0].eval()
            optimizer = None
        else:
            gpt_model, optimizer, _ = setup_model_and_optimizer(
                ModelType.encoder_or_decoder,
                self.model_provider,
                cfg_container=cfg_container,
                pg_collection=pg_collection,
            )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        # Hard coded to use cuda_graph_impl="transformer_engine"
        cuda_graph_impl = "transformer_engine"
        if use_cuda_graph and cuda_graph_impl == "transformer_engine":
            from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        num_fp8_params = 0
        for _, param in gpt_model[0].named_parameters():
            if not inference:
                assert param.requires_grad
                assert param.main_grad is not None
            if is_float8tensor(param):
                num_fp8_params += 1

        fp8_layers = args.num_layers
        if kwargs.get("first_last_layers_bf16", False):
            fp8_layers -= kwargs["num_layers_at_start_in_bf16"]
            fp8_layers -= kwargs["num_layers_at_end_in_bf16"]
        if fp8_param_gather and fp8_layers > 0:
            if args.num_experts is None:
                # Each dense layer has 4 GEMM weights: qkv, proj, fc1, fc2.
                assert num_fp8_params == 4 * fp8_layers
            else:
                assert num_fp8_params > 0
                assert any(
                    not getattr(param, 'allreduce', True) for param in gpt_model[0].parameters()
                )
                if not inference:
                    assert len(optimizer.chained_optimizers) >= 2

        # Verify that bf16 params (embedding, LN, etc.) in the MXFP8 model are mapped
        # to the param buffer (shared with grad buffer) rather than allocated separately.
        if args.reuse_grad_buf_for_mxfp8_param_ag:
            for buffer in gpt_model[0].buffers:
                if buffer.param_data is None:
                    continue
                buf_start = buffer.param_data.data_ptr()
                buf_end = buf_start + buffer.param_data.numel() * buffer.param_data.element_size()
                for param in buffer.param_to_bucket:
                    if is_mxfp8tensor(param):
                        # MXFP8 params keep their own quantized storage.
                        assert not (
                            buf_start <= param.data.data_ptr() < buf_end
                        ), "MXFP8 param should not be mapped to the param buffer"
                    else:
                        # BF16 params should be views into the param buffer
                        # (no double allocation).
                        assert buf_start <= param.data.data_ptr() < buf_end, (
                            "BF16 param should be a view into the param buffer "
                            "(no separate allocation)"
                        )

        loss_list = []
        eval_loss_list = []

        # Optional: generate the sharded_state_dict (the checkpoint-save metadata path) at these
        # steps to catch save side-effects on the live weights — a correct save must not perturb
        # the subsequent training step (regression guard for GTP native-FP8 save corruption).
        save_at_steps = set(save_at_steps_kw or ())

        for i in range(100):
            if not inference:
                gpt_model[0].zero_grad_buffer()
                optimizer.zero_grad()

            if i in save_at_steps:
                # Mirror production save_checkpoint_and_time: when the forward pre-hook is disabled
                # for the save, a forced param-sync runs first. Passing the optimizer makes it copy
                # the FP32 masters into the param buffer before the copy-back re-quantizes, so
                # native-FP8 GTP shards are refreshed from masters (not stale grad scratch).
                # Exercise it so the save-perturbation test is a real regression test for the
                # post-save loss spike.
                if should_disable_forward_pre_hook(args):
                    force_param_sync(gpt_model, optimizer=optimizer)
                _ = gpt_model[0].sharded_state_dict()

            # Capture CUDA graphs after warmup if helper is provided.
            # Hard coded cuda_graph_warmup_steps = 0.
            cuda_graph_warmup_steps = 0
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(gpt_model, param_sync=False)
                self.cuda_graph_helper.create_cudagraphs()
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(gpt_model)
                    self.cuda_graph_helper.cuda_graph_set_manual_hooks()

            # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
            # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
            # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                self.copy_main_params_to_param_buffer(gpt_model, optimizer)

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length

            if inference:
                continue

            # Verify gradients
            loss = output.mean()
            loss.backward()

            if args.overlap_grad_reduce:
                gpt_model[0].finish_grad_sync()

            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

            if eval_transition:
                eval_loss_list.append(
                    self.run_eval_transition(
                        args,
                        gpt_model,
                        optimizer,
                        (input_ids, labels, position_ids, attention_mask, loss_mask),
                    )
                )

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        if eval_transition:
            return torch.tensor(loss_list), torch.tensor(eval_loss_list)
        return torch.tensor(loss_list)

    def run_test(self, tp_size, recipe, inference: bool = False, **kwargs):
        """Test fp8_param with a small GPT model."""
        if inference:
            with torch.inference_mode():
                self._run_test_helper(tp_size, recipe, inference=True, **kwargs)
        else:
            loss_list = self._run_test_helper(tp_size, recipe, fp8_param_gather=True, **kwargs)

            # Before TE 2.2.0, we cannot guarantee that the main params are the same with/without
            # fp8-param-gather, so skip the checking of tensor values.
            if is_te_min_version("2.2.0"):
                loss_list_ref = self._run_test_helper(
                    tp_size, recipe, fp8_param_gather=False, **kwargs
                )
                torch.testing.assert_close(loss_list, loss_list_ref, atol=1e-4, rtol=1e-4)

    def run_test_with_cuda_graph(self, tp_size, recipe, **kwargs):
        loss = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, use_cuda_graph=True, **kwargs
        )
        loss_ref = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, use_cuda_graph=False, **kwargs
        )
        torch.testing.assert_close(loss, loss_ref, atol=0, rtol=0)

    def run_test_with_eval_transition(self, tp_size, recipe, **kwargs):
        loss, eval_loss = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, eval_transition=True, **kwargs
        )
        loss_ref, eval_loss_ref = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=False, eval_transition=True, **kwargs
        )
        torch.testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(eval_loss, eval_loss_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_delayed_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="delayed", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_delayed_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "delayed", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_tensorwise_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="tensorwise", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_tensorwise_scaling_inference(self, tp_size):
        self.run_test(tp_size=tp_size, recipe="tensorwise", inference=True)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_tensorwise_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "tensorwise", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_tensorwise_scaling_with_first_last_layers_bf16(self, tp_size):
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, recipe="tensorwise", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_blockwise_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="blockwise")

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_blockwise_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "blockwise", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    def test_mxfp8(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not mxfp8_2d_available, reason=reason_for_no_mxfp8_2d)
    def test_mxfp8_2d_dequantize_matches_reference(self):
        """2D MXFP8 dequantization must apply the scale of each 32x32 block."""
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        block_size = 32
        fp8_max = 448.0

        def float_to_e8m0(amax):
            scale = (amax.float() / fp8_max).contiguous()
            scale_u32 = scale.view(torch.int32)
            exponent = ((scale_u32 >> 23) & 0xFF).to(torch.int32)
            mantissa = scale_u32 & 0x7FFFFF
            round_up = (
                (mantissa > 0) & (exponent != 254) & ~((exponent == 0) & (mantissa <= 0x400000))
            )
            exponent = exponent + round_up.to(torch.int32)
            return torch.where(scale == 0, torch.zeros_like(exponent), exponent).to(torch.uint8)

        for shape in ((1280, 10240), (3072, 10240), (1056, 256)):
            rows, cols = shape
            generator = torch.Generator(device="cuda")
            generator.manual_seed(_SEED + rows)
            source = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
            # Make neighboring block scales observably different.
            source[0, 0] = 64.0
            source[block_size, block_size] = -16.0

            quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
                with_2d_quantization=True,
            )
            quantized = quantizer(source)

            block_rows = rows // block_size
            block_cols = cols // block_size
            source_blocks = source.view(block_rows, block_size, block_cols, block_size).permute(
                0, 2, 1, 3
            )
            block_amax = source_blocks.float().abs().amax(dim=(-1, -2))
            scale_bytes = float_to_e8m0(block_amax)
            block_scale = torch.pow(2.0, scale_bytes.float() - 127)
            scale = block_scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
            reference_data = (source.float() / scale).to(torch.float8_e4m3fn)
            reference = (reference_data.float() * scale).to(source.dtype)

            actual = quantized.dequantize()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(
                quantized._rowwise_data.view(torch.uint8),
                reference_data.view(torch.uint8),
                rtol=0,
                atol=0,
            )

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not mxfp8_2d_available, reason=reason_for_no_mxfp8_2d)
    def test_mxfp8_2d_adam_master_reload_with_gtp(self):
        """Adam masters must support both high-precision and forced-dequant reload paths."""
        args = self.create_test_args(
            2,
            "mxfp8",
            self.seq_length,
            self.micro_batch_size,
            False,
            True,
            False,
            mxfp8_2d_quantization=True,
            optimizer="adam",
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            tensor_parallel_num_weight_shards=4,
            global_batch_size=4,
        )
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, gtp_remat_size=args.gtp_weight_remat_size
        )
        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        assert args.gtp_weight_remat_size == 2
        assert len(model) == 1
        high_precision_state = {}
        for idx, (name, param) in enumerate(model[0].named_parameters()):
            if is_mxfp8tensor(param):
                value = dequantize_gtp_native_fp8(param).detach().clone()
                value.add_(
                    torch.tensor((idx % 7 + 1) / 128, dtype=value.dtype, device=value.device)
                )
            else:
                value = param.detach().clone()
            high_precision_state[name] = value

        tested_params = 0
        for optim_instance in optimizer.chained_optimizers:
            if not isinstance(optim_instance, DistributedOptimizer):
                continue
            state_map = optim_instance._build_model_param_to_state_dict_param_map(
                high_precision_state
            )
            expected_high_precision = []
            expected_dequantized = []
            for model_group, main_group in zip(
                optim_instance.model_float16_groups, optim_instance.shard_fp32_from_float16_groups
            ):
                for model_param, main_param in zip(model_group, main_group):
                    if not is_mxfp8tensor(model_param):
                        continue
                    assert model_param._quantizer.with_2d_quantization
                    assert getattr(model_param, "gtp_remat_size", 1) == 2
                    param_range = optim_instance._get_model_param_range_map(model_param)["param"]
                    expected_high_precision.append(
                        (
                            main_param,
                            state_map[model_param]
                            .view(-1)[param_range.start : param_range.end]
                            .float()
                            .clone(),
                        )
                    )
                    expected_dequantized.append(
                        (
                            main_param,
                            dequantize_gtp_native_fp8(model_param)
                            .view(-1)[param_range.start : param_range.end]
                            .float()
                            .clone(),
                        )
                    )
                    tested_params += 1

            assert expected_high_precision
            for main_param, _ in expected_high_precision:
                main_param.data.fill_(float("nan"))
            optim_instance.reload_model_params(state_dict=high_precision_state)
            for main_param, expected in expected_high_precision:
                torch.testing.assert_close(main_param, expected, rtol=0, atol=0)

            for main_param, _ in expected_dequantized:
                main_param.data.fill_(float("nan"))
            optim_instance.reload_model_params()
            for main_param, expected in expected_dequantized:
                torch.testing.assert_close(main_param, expected, rtol=0, atol=0)

        assert tested_params > 0

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [1])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    def test_mxfp8_moe(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {
            "overlap_param_gather": dp_overlap[0],
            "overlap_grad_reduce": dp_overlap[1],
            "num_layers": 4,
            "vocal_size": 128800,
            "hidden_size": 128,
            "num_attention_heads": 8,
            "expert_model_parallel_size": 2,
            "num_experts": 2,
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_topk": 1,
            "moe_router_pre_softmax": True,
            "moe_router_load_balancing_type": "none",
            "moe_aux_loss_coeff": 0.0,
            "moe_ffn_hidden_size": 128,
        }
        self.run_test(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_mxfp8_eval_transition(self, tp_size):
        kwargs = {"overlap_param_gather": True, "overlap_grad_reduce": True}
        self.run_test_with_eval_transition(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_mxfp8_with_cuda_graph(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_blockwise_scaling_with_first_last_layers_bf16(self, tp_size):
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, recipe="blockwise", **kwargs)
