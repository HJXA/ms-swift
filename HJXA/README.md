# swift 环境配置

```bash
git clone git@github.com:HJXA/ms-swift.git
uv venv .swift --python 3.12
source .swift/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # H200
uv pip install torch torchvision torchaudio   --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/uv # 镜像
uv pip install -e '.[all]'

uv pip install -e '.[all]' --index-url https://mirrors.aliyun.com/pypi/simple/

# 其他依赖
uv pip install https://wheels.vllm.ai/88d34c6409e9fb3c7b8ca0c04756f061d2099eb1/vllm-0.20.0%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl # vllm

uv pip install deepspeed liger-kernel swanlab nvitop 
uv pip install deepspeed liger-kernel swanlab nvitop --index-url https://mirrors.aliyun.com/pypi/simple/

uv pip install qwen_vl_utils qwen_omni_utils keye_vl_utils pre-commit math_verify py-spy wandb
uv pip install qwen_vl_utils qwen_omni_utils keye_vl_utils pre-commit math_verify py-spy wandb --index-url https://mirrors.aliyun.com/pypi/simple/


# flash_attn

# 自建包 https://github.com/Dao-AILab/flash-attention/issues/2425

python --version &&
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" &&
nvcc -V

uv pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" # python 3.12 torch 2.11
uv pip install xxx.whl

```