# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.tensor_parallel import (gather_from_sequence_parallel_region,
                                           reduce_scatter_to_sequence_parallel_region)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from typing import Optional

from swift.model import ModelType
from swift.template import Template
from ..constant import MegatronModelType
from ..gpt_bridge import MultimodalGPTBridge
from ..modules import GatedSelfAttention
from ..register import MegatronModelLoader, MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Qwen3_5Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
        super().__init__(config, [Qwen3_5TextModel, Qwen3_5MoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.hf_config)


class Qwen3_5Bridge(MultimodalGPTBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        is_linear_attention = (layer_idx + 1) % self.config.linear_attention_freq != 0
        if is_linear_attention:
            hf_state_dict.update(
                self._set_linear_attn_state(mg_attn, hf_state_dict, 'linear_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.in_proj.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
        return hf_state_dict


class Qwen3_5Loader(MegatronModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        layer_specs = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in layer_specs.layer_specs:
            attn_module = layer_spec.submodules.self_attention.module
            if issubclass(attn_module, SelfAttention):
                layer_spec.submodules.self_attention.module = GatedSelfAttention
        return layer_specs

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ):
        model = super().build_model(pre_process, post_process, vp_stage)
        for layer in model.language_model.decoder.layers:
            if hasattr(layer.self_attention, 'out_norm'):
                assert hasattr(layer.self_attention.out_norm, 'zero_centered_gamma')
                layer.self_attention.out_norm.zero_centered_gamma = False
        return model


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_5,
        [
            ModelType.qwen3_5,
            ModelType.qwen3_5_moe,
        ],
        bridge_cls=Qwen3_5Bridge,
        visual_cls=Qwen3_5Vit,
        loader=Qwen3_5Loader,
    ))
