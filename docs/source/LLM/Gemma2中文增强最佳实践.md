
# Gemma2中文增强最佳实践

gemma2是google提出的最新的大语言模型, 包括gemma2-9b, gemma2-9b-it, gemma2-27b, gemma2-27b-it四个模型. 该文档将展示对gemma2-9b-instruct使用经过清洗的中英文SFT通用、代码和数学数据集进行中文增强，并使用自我认知数据集修改模型对自己和作者的认知。

我们使用的数据集链接如下：
SFT数据集：https://modelscope.cn/datasets/swift/swift-sft-mixture
自我认知数据集：https://modelscope.cn/datasets/swift/self-cognition

## 目录
- [环境准备](#环境准备)
- [原始模型](#原始模型)
- [中文增强微调](#中文增强微调)
- [微调后模型](#微调后模型)

## 环境准备

```shell
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
# 安装评测相关依赖
pip install -e '.[eval]'

# gemma2依赖
pip install transformers>=4.42

# 如果要使用vllm对gemma2进行推理加速, 需要使用源代码方式进行安装 (or vllm>=0.5.1)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

## 原始模型

### 推理
```shell
# Experimental environment: A100
# 如果是本地模型, 需指定`--model_type gemma2-9b-instruct --model_id_or_path <local_path>`
# 如果使用原生pytorch进行推理, 请设置`--infer_backend pt`
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_id_or_path LLM-Research/gemma-2-9b-it \
    --infer_backend vllm
```

我们使用几个样例进行测试，发现其中文能力并不稳定：
```python
"""
<<< 你是谁？
我是一个大型语言模型，由 Google 训练。

我的目标是帮助人们通过理解和生成文本来完成各种任务。我可以回答问题、总结文本、创作故事、翻译语言等等。

我仍然在学习中，但我很乐意尽力帮助你！
--------------------------------------------------
<<< what's your name?
I don't have a name. My creators, the Gemma team, haven't given me one. You can just call me Gemma!
--------------------------------------------------
<<< 你是谁研发的？
我由 Google DeepMind 研发。 😊 

虽然我是一个大型语言模型，但我的开发和维护是由 Google DeepMind 的工程师和研究人员完成的。
--------------------------------------------------
<<< 浙江的省会在哪？
浙江的省会是 **杭州**。  😊 



--------------------------------------------------
<<< 这有什么好吃的？
杭州美食众多，真是让人垂涎三尺！ 🤤

以下是一些杭州的特色美食：

* **西湖醋鱼:** 杭州的代表菜，鱼肉鲜嫩，醋汁酸甜开胃，是必尝之选。
* **龙井茶:** 杭州名茶，清香甘醇，是品茶爱好者的首选。
* **杭州小笼包:** 皮薄馅多，汤汁鲜美，是杭州人早餐的常客。
* **豆腐脑:**  杭州人早餐的另一选择，配上香油、辣椒油和葱花，味道鲜美。
* **绍兴酒:**  浙江绍兴产的著名黄酒，香醇甘甜，是宴席上的佳酿。

还有很多其他美味佳肴，比如：

* **东坡肉:**  以其肥而不腻、香甜可口的口感著称。
* **喵呜鱼:**  以其鲜嫩的鱼肉和独特的酱汁闻名。
* **清汤面:**  杭州人日常生活中常见的美食，汤头清澈，面条劲道。

如果你有机会去杭州旅行，一定要品尝一下这些美味佳肴！



--------------------------------------------------
"""
```

### 评测

```shell
# Experimental environment: A100
# 评测后端由llmuses库提供: https://github.com/modelscope/eval-scope
# 推荐使用vllm进行推理加速. 如果使用原生pytorch进行推理, 请设置`--infer_backend pt`
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model_id_or_path LLM-Research/gemma-2-9b-it \
    --eval_dataset arc ceval gsm8k mmlu --eval_backend Native \
    --infer_backend vllm
```


| Model              | arc    | ceval  | gsm8k  | mmlu   |
| ------------------ | ------ | ------ | ------ | ------ |
| llama3-8b-instruct | 0.7628 | 0.5111 | 0.7475 | 0.6369 |
| gemma2-9b-instruct | 0.8797 | 0.5275 | 0.8143 | 0.627  |



## 中文增强微调
这里为了降低训练的时间，对数据集进行了较少的采样。如果要想进行更多数据集的微调，可以适当增大混合的比例，例如：`--dataset swift-mix:sharegpt#50000 swift-mix:firefly#20000 swift-mix:codefuse#20000 swift-mix:metamathqa#20000 self-cognition#1000`。

我们对embedding层、所有的linear层和lm_head层加上lora，并设置layer_norm层可训练。我们在中文增强的同时，修改模型的自我认知，让模型认为自己是小黄，由魔搭创建。

```shell
# Experimental environment: 4 * A100
# 4 * 80GB GPU memory
# # 如果是本地模型, 需指定`--model_type gemma2-9b-instruct --model_id_or_path <local_path>`
MASTER_PORT=29500 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft  \
    --model_id_or_path LLM-Research/gemma-2-9b-it \
    --dataset swift-mix:sharegpt#10000 swift-mix:firefly#5000 swift-mix#5000 swift-mix:metamathqa#5000 self-cognition#500 \
    --lora_target_modules EMBEDDING ALL lm_head \
    --lora_modules_to_save LN \
    --adam_beta2 0.95 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --eval_steps 100 \
    --max_length 8192 \
    --gradient_accumulation_steps 16 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --save_total_limit -1 \
    --logging_steps 5 \
    --use_flash_attn true \
```


## 微调后模型

### 推理
```shell
# Experimental environment: A100
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/gemma2-9b-instruct/vx-xxx/checkpoint-xxx \
    --infer_backend vllm --merge_lora true
```


### 评测
```shell
# Experimental environment: A100
CUDA_VISIBLE_DEVICES=0 swift eval \
    --ckpt_dir output/gemma2-9b-instruct/vx-xxx/checkpoint-xxx \
    --eval_dataset arc ceval gsm8k mmlu --eval_backend Native \
    --infer_backend vllm --merge_lora true
```

