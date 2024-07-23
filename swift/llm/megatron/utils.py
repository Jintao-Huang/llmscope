import os
import sys
from functools import partial, wraps
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from swift.llm import LazyLLMDataset, Template, git_clone_github, is_megatron_available
from swift.utils import append_to_jsonl, get_dist_setting, get_logger, is_master, subprocess_run

logger = get_logger()


def init_megatron_env() -> None:

    if 'MEGATRON_LM_PATH' not in os.environ:
        megatron_path = git_clone_github('https://github.com/NVIDIA/Megatron-LM')
        os.environ['MEGATRON_LM_PATH'] = megatron_path
    else:
        megatron_path = os.environ['MEGATRON_LM_PATH']
    if not is_megatron_available():
        subprocess_run(['pip', 'install', '-e', megatron_path])
    sys.path.append(megatron_path)

    if 'PAI_MEGATRON_PATCH_PATH' not in os.environ:
        megatron_patch_path = git_clone_github('https://github.com/alibaba/Pai-Megatron-Patch')
        os.environ['PAI_MEGATRON_PATCH_PATH'] = megatron_patch_path
    sys.path.append(os.environ['PAI_MEGATRON_PATCH_PATH'])


def get_model_seires(model_type: str) -> str:
    if model_type.startswith('qwen2'):
        return 'qwen2'
    else:
        raise ValueError(f'model_type: {model_type} not supported')


def patch_megatron(tokenizer):

    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    from megatron.training import get_args, training, initialize, global_vars
    global_vars.build_tokenizer = build_tokenizer

    _old_initialize_distributed = initialize._initialize_distributed

    @wraps(_old_initialize_distributed)
    def _initialize_distributed():
        args = get_args()
        if dist.is_initialized():
            args.rank, args.local_rank, args.world_size, args.local_world_size = get_dist_setting()
            torch.cuda.set_device(args.local_rank)
        return _old_initialize_distributed()

    initialize._initialize_distributed = _initialize_distributed

    _old_load_checkpoint = training.load_checkpoint

    @wraps(_old_load_checkpoint)
    def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=False):
        # default: strict=False
        return _old_load_checkpoint(model, optimizer, opt_param_scheduler, load_arg=load_arg, strict=strict)

    training.load_checkpoint = load_checkpoint

    _old_training_log = training.training_log

    @wraps(_old_training_log)
    def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                     report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad):
        args = get_args()
        if is_master() and iteration % args.log_interval == 0:
            logging_path = os.path.join(args.save, 'logging.jsonl')
            logs = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                logs[k] = round(v, 8)
            logs['grad_norm'] = round(grad_norm, 8)
            logs['learning_rate'] = round(learning_rate, 8)
            logs['consumed_samples'] = args.consumed_train_samples
            logs['global_step/max_steps'] = f'{iteration}/{args.train_iters}'
            append_to_jsonl(logging_path, logs)
        return _old_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                                 loss_scale, report_memory_flag, skipped_iter, grad_norm, params_norm,
                                 num_zeros_in_grad)

    training.training_log = training_log


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function. copy from Pai-Megatron-Patch

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    from megatron.training import get_args
    from megatron.core import mpu
    from megatron.training.utils import average_losses_across_data_parallel_group
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                  f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples, train_dataset: LazyLLMDataset,
                                       val_dataset: LazyLLMDataset, template: Template):
    # train_val_test_num_samples: ignored
    from megatron.training import training
    from megatron.training.utils import get_ltor_masks_and_position_ids

    assert not hasattr(training, '_old_build_pretraining_data_loader')
    _old_build_pretraining_data_loader = training.build_pretraining_data_loader

    def data_collator(batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = template.data_collator(batch, padding_to)
        labels = res['labels']
        new_labels = torch.zeros_like(labels)
        new_labels[:, :-1] = labels[:, 1:]
        new_labels[:, -1] = -100
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(new_labels, -100, False, False, True)
        return {
            'tokens': res['input_ids'],
            'labels': new_labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }

    @wraps(_old_build_pretraining_data_loader)
    def build_pretraining_data_loader(*args, **kwargs):
        res = _old_build_pretraining_data_loader(*args, **kwargs)
        if res is not None:
            res.collate_fn = data_collator
        return res

    training.build_pretraining_data_loader = build_pretraining_data_loader
    training._old_build_pretraining_data_loader = _old_build_pretraining_data_loader
    return train_dataset, val_dataset, None
