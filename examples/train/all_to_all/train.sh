# 40GiB * 2
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
max_position_embeddings=10240 \
image_area=518400 \
swift sft \
    --model BAAI/Emu3-Gen \
    --train_type lora \
    --dataset 'swift/TextCaps#40' \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 0 \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 10000 \
    --weight_decay 0.1 \
    --deepspeed zero2
