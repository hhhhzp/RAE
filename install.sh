conda create -n rae python=3.10 -y
conda activate rae
pip install uv


unset https_proxy
unset http_proxy
# Install PyTorch 2.8.0 with CUDA 12.9 # or your own cuda version
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://mirrors.cloud.tencent.com/pypi/simple

# Install other dependencies
uv pip install -r requirements.txt --index-url https://mirrors.cloud.tencent.com/pypi/simple