# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install vllm -U
pip install lmdeploy -U --no-deps
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes -U
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
pip install timm -U
pip install deepspeed==0.14.*
pip install qwen_vl_utils decord librosa pyav icecream -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
