# Copyright (c) Alibaba, Inc. and its affiliates.
import sys

from transformers import AutoModel

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import git_clone_github, safe_snapshot_download


def get_model_tokenizer_got_ocr2(*args, **kwargs):
    kwargs['automodel_class'] = AutoModel
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2, [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR2_0', 'stepfun-ai/GOT-OCR2_0'),
            ]),
        ],
        TemplateType.got_ocr2,
        get_model_tokenizer_got_ocr2,
        model_arch=ModelArch.got_ocr2,
        architectures=['GOTQwenForCausalLM'],
        tags=['vision']))


def get_model_tokenizer_step_audio(*args, **kwargs):
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/stepfun-ai/Step-Audio.git')
    sys.path.append(local_repo_path)
    from tokenizer import StepAudioTokenizer
    from tts import StepAudioTTS
    encoder_path = safe_snapshot_download('stepfun-ai/Step-Audio-Tokenizer')
    decoder_path = safe_snapshot_download('stepfun-ai/Step-Audio-TTS-3B')
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    model.encoder = StepAudioTokenizer(encoder_path)
    model.decoder = StepAudioTTS(decoder_path, model.encoder)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.step_audio, [
            ModelGroup([
                Model('stepfun-ai/Step-Audio-Chat', 'stepfun-ai/Step-Audio-Chat'),
            ]),
        ],
        TemplateType.step_audio,
        get_model_tokenizer_step_audio,
        model_arch=ModelArch.step_audio,
        architectures=['GOTQwenForCausalLM'],
        tags=['vision']))
