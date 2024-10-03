# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.trainers import TrainerFactory
from swift.utils import get_logger, get_main, seed_everything
from ..argument import RLHFArguments
from ..template import TEMPLATE_MAPPING
from .patcher import TrainTemplate
from .sft import prepare_dataset, prepare_train_model_template, trainer_train

logger = get_logger()


def llm_rlhf(args: RLHFArguments) -> Dict[str, Any]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)

    is_generation = TEMPLATE_MAPPING[args.template_type].get('is_generation', False)
    if is_generation:
        logger.warning(f"Please check if args.template_type: '{args.template_type}' is correct.")

    msg = {}
    model, ref_model, template, callbacks, optimizers = prepare_train_model_template(args)
    with TrainerFactory.patch_template(args, template):
        template = TrainTemplate(template)
        train_dataset, val_dataset = prepare_dataset(args, template, msg)

        return trainer_train(
            args,
            model,
            template,
            train_dataset,
            val_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            msg=msg,
            ref_model=ref_model)


rlhf_main = get_main(RLHFArguments, llm_rlhf)
