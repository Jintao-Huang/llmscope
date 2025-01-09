# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Callable, List, Optional

from swift.llm import ModelGroup
from swift.llm.model.register import _get_matched_model_meta

MEGATRON_MODEL_MAPPING = {}


@dataclass
class MegatronModelMeta:
    megatron_model_type: Optional[str]
    model_groups: List[ModelGroup]

    convert_megatron2hf: Callable
    convert_hf2megatron: Callable
    get_model_provider: Callable
    load_config: Callable


def register_megatron_model(model_meta: MegatronModelMeta, *, exist_ok: bool = False):
    megatron_model_type = model_meta.megatron_model_type
    if not exist_ok and megatron_model_type in MEGATRON_MODEL_MAPPING:
        raise ValueError(f'The `{megatron_model_type}` has already been registered in the MODEL_MAPPING.')

    MEGATRON_MODEL_MAPPING[megatron_model_type] = model_meta


def get_megatron_model_meta(model_id_or_path: str) -> Optional[MegatronModelMeta]:
    return _get_matched_model_meta(model_id_or_path, MEGATRON_MODEL_MAPPING)
