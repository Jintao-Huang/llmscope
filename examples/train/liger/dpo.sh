# test env: 4 * A100
# Using use_liger_kernel: 4 * 61GB, 4 hour 30 minutes
# Not using use_liger_kernel: 4 * 54GB, 1 hour 40 minutes

nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'swift/RLAIF-V-Dataset#20000' \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_only_model true \
    --deepspeed zero3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --use_liger_kernel true
