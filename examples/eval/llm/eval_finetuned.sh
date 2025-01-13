CUDA_VISIBLE_DEVICES=0 \
swift eval \
  --adapters output/vx-xxx/checkpoint-xxx \
  --infer_backend vllm \
  --eval_limit 100 \
  --eval_dataset gsm8k
