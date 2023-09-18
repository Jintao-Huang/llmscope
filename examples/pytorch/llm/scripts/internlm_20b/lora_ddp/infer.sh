CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type internlm-20b \
    --sft_type lora \
    --template_type default-generation \
    --dtype bf16 \
    --ckpt_dir "output/internlm-20b/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset jd-zh \
    --max_length 2048 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
