# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="resources/banner.png"/>
    <br>
<p>
<p align="center">
<a href="https://modelscope.cn/home">魔搭社区官网</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>


<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.17-5D91D4.svg"></a>
<a href="https://pypi.org/project/ms-swift/"><img src="https://badge.fury.io/py/ms-swift.svg"></a>
<a href="https://github.com/modelscope/swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/swift"></a>
<a href="https://pepy.tech/project/ms-swift"><img src="https://pepy.tech/badge/ms-swift"></a>
<a href="https://github.com/modelscope/swift/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/6427" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6427" alt="modelscope%2Fswift | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
        <a href="https://arxiv.org/abs/2408.05517">论文</a> &nbsp ｜ <a href="https://swift.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://swift.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p>
<p align="center">
        <a href="https://swift2x-en.readthedocs.io/en/latest/">Swift2.x En Doc</a> &nbsp ｜ &nbsp <a href="https://swift2x.readthedocs.io/zh-cn/latest/">Swift2.x中文文档</a> &nbsp
</p>


##  📖 目录
- [简介](#-简介)
- [用户群](#-用户群)
- [新闻](#-新闻)
- [快速开始](#-快速开始)
- [License](#-license)
- [引用](#-引用)

## 📝 简介
🍲 ms-swift是魔搭社区官方提供的LLM与多模态LLM微调部署框架，现已支持400+LLM与100+多模态LLM的训练（预训练、微调、人类对齐）、推理、评测、量化与部署。其中LLM包括：Qwen2.5、Llama3.2、GLM4、Internlm2.5、Yi1.5、Mistral、DeepSeek、Baichuan2、Gemma2、TeleChat2等模型，多模态LLM包括：Qwen2-VL、Qwen2-Audio、Llama3.2-Vision、Llava、InternVL2、MiniCPM-V-2.6、GLM4v、Xcomposer2.5、Yi-VL、DeepSeek-VL、Phi3.5-Vision等模型。

🍔 除此之外，ms-swift汇集了最新的训练技术，包括LoRA、QLoRA、Llama-Pro、LongLoRA、GaLore、Q-GaLore、LoRA+、LISA、DoRA、FourierFt、ReFT、UnSloth、Megatron和Liger等。ms-swift支持使用vLLM和LMDeploy对推理、评测和部署模块进行加速。为了帮助研究者和开发者更轻松地微调和应用大模型，ms-swift还提供了基于Gradio的Web-UI界面及丰富的最佳实践。

## ☎ 用户群

请扫描下面的二维码来加入我们的交流群：

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  微信群
:-------------------------:|:-------------------------:
<img src="asset/discord_qr.jpg" width="200" height="200">  |  <img src="asset/wechat.png" width="200" height="200">


## 🎉 新闻
- 🎁2024.12.04: SWIFT3.0大版本更新. 请查看[ReleaseNote和BreakChange](./docs/source/Instruction/ReleaseNote3.0.md).
- 🔥2024.09.19: 支持Qwen2.5、Qwen2-VL、Qwen2-Audio系列模型.
- 🔥2024.08.20: 支持使用deepspeed zero2/zero3对多模态模型进行预训练、微调和RLHF.
- 🔥2024.08.12: 🎉SWIFT论文已经发布到arXiv上，可以点击[这个链接](https://arxiv.org/abs/2408.05517)阅读.
- 🔥2024.08.05: 支持使用[evalscope](https://github.com/modelscope/evalscope/)作为后端进行多模态模型的评测.
- 🔥2024.07.29: 支持使用[vllm](https://github.com/vllm-project/vllm), [lmdeploy](https://github.com/InternLM/lmdeploy)对大模型和多模态大模型进行推理加速，在infer/deploy/eval时额外指定`--infer_backend vllm/lmdeploy`即可.
- 🔥2024.07.24: 人类偏好对齐算法DPO/ORPO/SimPO/CPO/KTO/RM支持多模态大模型.
- 🔥2024.02.01: 支持Agent训练！Agent训练算法源自这篇[论文](https://arxiv.org/pdf/2309.00986.pdf). 我们也增加了[ms-agent](https://www.modelscope.cn/datasets/iic/ms_agent/summary)这个优质的agent数据集. 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora/sft.sh)开启Agent训练!

## 🛠️ 安装
使用pip进行安装：
```shell
pip install ms-swift -U
```

从源代码安装：
```shell
# pip install git+https://github.com/modelscope/ms-swift.git

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

## 🚀 快速开始

本章节介绍基本使用，更丰富的使用方式请查看[文档部分](https://swift.readthedocs.io/zh-cn/latest/)。

### Web-UI

Web-UI是基于gradio界面技术的**零门槛**训练部署界面方案。Web-UI配置简单，且完美支持多卡训练和部署：

```shell
swift web-ui
```
![image.png](./docs/resources/web-ui.png)

### 命令行

#### 训练
```shell
swift sft \
    --model <model_id_or_path> \
    --dataset <dataset_id_or_path> \
    --train_type lora \
    --output_dir output \
    ...
```

#### RLHF
```shell
swift sft \
    --model <model_id_or_path> \
    --dataset <dataset_id_or_path> \
    --train_type lora \
    --rlhf_type lora \
    --output_dir output \
    ...
```

#### 推理


#### 部署


#### 评测


### 使用Python



## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。

## 📎 引用

```bibtex
@misc{zhao2024swiftascalablelightweightinfrastructure,
      title={SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning},
      author={Yuze Zhao and Jintao Huang and Jinghan Hu and Xingjun Wang and Yunlin Mao and Daoze Zhang and Zeyinzi Jiang and Zhikai Wu and Baole Ai and Ang Wang and Wenmeng Zhou and Yingda Chen},
      year={2024},
      eprint={2408.05517},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.05517},
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/ms-swift&Date)
