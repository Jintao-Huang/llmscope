# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch.distributed as dist

from swift.utils import get_logger, is_dist
from .infer_args import InferArguments

logger = get_logger()


@dataclass
class ExportArguments(InferArguments):
    to_peft_format: bool = False
    to_ollama: bool = False
    ollama_output_dir: Optional[str] = None
    gguf_file: Optional[str] = None

    # awq: 4; gptq: 2, 3, 4, 8
    quant_bits: int = 0  # e.g. 4
    quant_method: Literal['awq', 'gptq', 'bnb'] = 'awq'
    quant_n_samples: int = 256
    quant_seqlen: Optional[int] = None  # use max_length
    quant_device_map: Optional[str] = None  # e.g. 'cpu', 'auto'
    quant_output_dir: Optional[str] = None
    quant_batch_size: int = 1

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    commit_message: str = 'update files'

    # megatron
    to_megatron: bool = False
    to_hf: bool = False
    megatron_output_dir: Optional[str] = None
    hf_output_dir: Optional[str] = None
    tp: int = 1
    pp: int = 1

    # The parameter has been defined in InferArguments.
    # merge_lora, hub_token

    def select_dtype(self) -> None:
        if self.quant_bits > 0 and self.dtype == 'auto':
            self.dtype = 'fp16'
            logger.info(f'Setting args.dtype: {self.dtype}')
        super().select_dtype()

    def handle_merge_device_map(self):
        if self.merge_device_map is None and self.quant_bits > 0:
            self.merge_device_map = 'cpu'

    def __post_init__(self):
        super().__post_init__()
        if self.quant_seqlen is None:
            self.quant_seqlen = self.max_length
        if self.quant_bits > 0:
            if len(self.dataset) == 0:
                self.dataset = ['alpaca-zh#10000', 'alpaca-en#10000']
                logger.info(f'Setting args.dataset: {self.dataset}')
            if self.quant_output_dir is None:
                if self.ckpt_dir is None:
                    self.quant_output_dir = f'{self.model_type}-{self.quant_method}-int{self.quant_bits}'
                else:
                    ckpt_dir, ckpt_name = os.path.split(self.ckpt_dir)
                    self.quant_output_dir = os.path.join(ckpt_dir,
                                                         f'{ckpt_name}-{self.quant_method}-int{self.quant_bits}')
                self.quant_output_dir = self.check_path_validity(self.quant_output_dir)
                logger.info(f'Setting args.quant_output_dir: {self.quant_output_dir}')
            assert not os.path.exists(self.quant_output_dir), f'args.quant_output_dir: {self.quant_output_dir}'
        elif self.to_ollama:
            assert self.sft_type in ['full'] + self.adapters_can_be_merged
            if self.sft_type in self.adapters_can_be_merged:
                self.merge_lora = True
            if not self.ollama_output_dir:
                self.ollama_output_dir = f'{self.model_type}-ollama'
            self.ollama_output_dir = self.check_path_validity(self.ollama_output_dir)
            assert not os.path.exists(
                self.ollama_output_dir), f'Please make sure your output dir does not exists: {self.ollama_output_dir}'
        elif self.to_megatron or self.to_hf:
            self.quant_method = None
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
            assert is_dist(), 'Please start in distributed mode.'
            dist.init_process_group(backend='nccl')
        if self.to_megatron:
            if self.megatron_output_dir is None:
                self.megatron_output_dir = f'{self.model_type}-tp{self.tp}-pp{self.pp}'
            self.megatron_output_dir = self.check_path_validity(self.megatron_output_dir)
            logger.info(f'Setting args.megatron_output_dir: {self.megatron_output_dir}')
        if self.to_hf:
            if self.hf_output_dir is None:
                self.hf_output_dir = os.path.join(self.ckpt_dir, f'{self.model_type}-hf')
            self.hf_output_dir = self.check_path_validity(self.hf_output_dir)
            logger.info(f'Setting args.hf_output_dir: {self.hf_output_dir}')
