conda create -n rae python=3.10 -y

unset https_proxy
unset http_proxy
conda run -n rae pip install uv
conda run -n rae uv pip install torch==2.8.0 torchvision torchaudio \
  --index-url https://mirrors.cloud.tencent.com/pypi/simple
conda run -n rae uv pip install -r requirements.txt \
  --index-url https://mirrors.cloud.tencent.com/pypi/simple
