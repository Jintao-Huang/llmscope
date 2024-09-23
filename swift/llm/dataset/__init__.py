# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import datasets.fingerprint
from swift.llm.dataset.preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor,
                                          ConversationsPreprocessor,
                                          ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor,
                                          SmartPreprocessor,
                                          TextGenerationPreprocessor)
from swift.utils.torch_utils import _find_local_mac


def _update_fingerprint_mac(*args, **kwargs):
    # Prevent different nodes use the same location in unique shared disk
    mac = _find_local_mac().replace(':', '')
    fp = datasets.fingerprint._update_fingerprint(*args, **kwargs)
    fp += '-' + mac
    if len(fp) > 64:
        fp = fp[:64]
    return fp


datasets.fingerprint._update_fingerprint = datasets.fingerprint.update_fingerprint
datasets.fingerprint.update_fingerprint = _update_fingerprint_mac
datasets.arrow_dataset.update_fingerprint = _update_fingerprint_mac
