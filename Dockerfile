# syntax=docker/dockerfile:1

FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# 1) Dependencias básicas y filtros FFmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      frei0r-plugins \
      build-essential \
      git \
      cmake \
      ninja-build \
      python3 \
      python3-pip \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2) PyTorch CPU/MPS, BasicSR y Real‑ESRGAN desde GitHub
RUN pip3 install --no-cache-dir \
      torch torchvision==0.14.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir git+https://github.com/xinntao/BasicSR.git@master && \
    pip3 install --no-cache-dir git+https://github.com/xinntao/Real-ESRGAN.git@master


# 3) Copiar tu script de mejora
WORKDIR /workspace
COPY enhance.sh /workspace/enhance.sh
RUN chmod +x /workspace/enhance.sh

ENTRYPOINT ["/workspace/enhance.sh"]
