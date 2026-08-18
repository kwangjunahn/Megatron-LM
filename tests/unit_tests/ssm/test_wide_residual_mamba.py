# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for streamwise wide-residual connections around Mamba mixers."""

import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.wide_residual_layer import (
    StreamwiseSigmoidWideResidualConnection,
    expand_wide_residual_stream,
    specialize_wide_residual_layer_spec,
)


class _AdditiveMixer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.last_hidden_size = None

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        self.last_hidden_size = hidden_states.shape[-1]
        bias = torch.ones(
            hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        return 2.0 * hidden_states, bias


def _config(*, wide: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
        wide_residual=(
            WideResidualConfig(num_streams=4, streamwise_sigmoid_init_scale=0.0) if wide else None
        ),
    )


def _layer_spec() -> ModuleSpec:
    return ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            norm=IdentityOp, mixer=ModuleSpec(module=_AdditiveMixer), mamba_bda=get_bias_dropout_add
        ),
    )


def _build_layer(config: TransformerConfig) -> MambaLayer:
    spec = _layer_spec()
    if config.wide_residual is not None:
        spec = specialize_wide_residual_layer_spec(spec, config)
    return build_module(spec, config=config, layer_number=1, pg_collection=ProcessGroupCollection())


def test_mamba_specialization_is_copy_on_write():
    original = _layer_spec()
    specialized = specialize_wide_residual_layer_spec(original, _config(wide=True))

    assert original.submodules.residual_connection is None
    assert (
        specialized.submodules.residual_connection.module is StreamwiseSigmoidWideResidualConnection
    )


def test_mamba_wide_connection_matches_ordinary_residual_at_initialization():
    base = torch.randn(2, 3, 8, requires_grad=True)
    baseline = _build_layer(_config(wide=False))
    wide = _build_layer(_config(wide=True))
    residual_stream = expand_wide_residual_stream(base, 4)

    baseline_output = baseline(base, attention_mask=None)
    wide_output = wide(residual_stream, attention_mask=None)

    assert wide.residual_connection is not None
    assert wide.residual_stream_hidden_size == 4 * wide.config.hidden_size
    assert wide.mixer.last_hidden_size == wide.config.hidden_size
    assert wide_output.shape[-1] == 4 * wide.config.hidden_size
    assert torch.allclose(wide_output, expand_wide_residual_stream(baseline_output, 4))

    wide_output.sum().backward()
    assert wide.residual_connection.read_map.logit.grad is not None
    assert wide.residual_connection.write_map.logit.grad is not None
