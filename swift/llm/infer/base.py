# Copyright (c) Alibaba, Inc. and its affiliates.

from types import MethodType
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from swift.llm.template.base import Context, Template, _findall


class InferTemplate:
    """
    This class is used in inference, and wraps the operations needed by vLLM and LMDeploy.
    """

    def __init__(self, template: Template, infer_backend: Literal['vllm', 'lmdeploy']):
        if infer_backend in {'vllm', 'lmdeploy'}:
            template.load_medias = True
            template._encode = MethodType(self._encode, template)
            template.check_example = MethodType(self.check_example, template)
            if framework == 'lmdeploy':
                template.image_placeholder = [[-100]]
        self.template = template
        self.framework = framework

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = Template._encode(self.template, example)
        if self.framework in ('vllm', 'lmdeploy'):
            inputs['images'] = example.get('images')
        return inputs, tokenizer_kwargs

    def check_example(self, example):
        if self.template.name in ('minicpm-v-v2_5', 'minicpm-v-v2_6', 'qwen-vl') and self.framework in ('vllm',
                                                                                                        'lmdeploy'):
            return
        return self.template.check_example(example)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        if media_type == 'image' and self.framework == 'lmdeploy':
            return [[-100]]
        if self.template.template_type == 'qwen-vl':
            if self.framework == 'lmdeploy':
                return [f'Picture {index + 1}: ', [-100], '\n']
            if self.framework == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
        if 'internvl' in self.template.template_type:
            if self.framework == 'vllm':
                return ['<img><image></img>\n']
        if self.template.template_type == 'llava-yi':
            if self.framework == 'vllm':
                return [[64000], '\n']
        if self.template.template_type == 'paligemma':
            if self.framework == 'vllm':
                self.template.prompt = ['{{QUERY}}']
                return []
        if self.template.template_type == 'phi3-vl':
            if self.framework == 'vllm':
                return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        if self.template.template_type in ('minicpm-v-v2_5', 'minicpm-v-v2_6'):
            if self.framework == 'vllm':
                return ['(<image>./</image>)\n']
        return self.template.replace_tag(media_type, index, example)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.template, name)

    async def _minicpm_v_prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        idx_list.insert(0, -1)
        new_input_ids = []
        features = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            context_list = ['<image>', [-100], '</image>']
            feat = [x.squeeze() for x in images[i]['embeddings'].split(1)]
            grid = images[i].get('grid')
            if len(feat) > 1 and grid is not None:
                context_list.append('<slice>')
                for j in range(grid[1]):
                    if j > 0:
                        context_list.append('\n')
                    for _ in range(grid[0]):
                        context_list += ['<image>', [-100], '</image>']
                context_list.append('</slice>\n')
            new_input_ids += self._encode_context_list(context_list)[0]
            features += feat
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['images'] = features

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        if self.template.template_type == ('minicpm-v-v2_5', 'minicpm-v-v2_6'):
            await self._minicpm_v_prepare_lmdeploy_inputs(inputs)
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        ranges = []
        for i in range(len(idx_list) - 1):
            _range = []
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            _range.append(len(new_input_ids))
            new_input_ids += [IMAGE_DUMMY_TOKEN_INDEX] * images[i].shape[0]
            _range.append(len(new_input_ids))
            ranges.append(_range)
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_embeddings'] = images
        inputs['input_embedding_ranges'] = ranges
        inputs['input_ids'] = new_input_ids


class InferEngine:

    def __init__(self, llm_engine, template):
        self.llm_engine = llm_engine
        self.template = template

    def infer(self,
              request_list: List[Dict[str, Any]],
              *,
              generation_config: Optional[Any] = None,
              generation_info: Optional[Dict[str, Any]] = None,
              max_batch_size: Optional[int] = None,
              lora_request: Optional[Any] = None,
              use_tqdm: bool = False,
              verbose: bool = False,
              prompt_prefix: str = '[PROMPT]',
              output_prefix: str = '[OUTPUT]',
              **kwargs) -> List[Dict[str, Any]]:
        pass

    def inference_stream(
            self,
            request_list: List[Dict[str, Any]],
            *,
            generation_config: Optional[Any] = None,
            generation_info: Optional[Dict[str, Any]] = None,
            lora_request: Optional['LoRARequest'] = None,
            use_tqdm: bool = False,
            flush_steps: Optional[int] = None,  # Ensuring efficiency
            **kwargs) -> Iterator[List[Dict[str, Any]]]:
        pass
