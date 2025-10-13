FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="fuliucansheng@gmail.com"
LABEL repository="unitorch"

ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   locales \
                   ca-certificates \
                   zip \
                   unzip \
                   ninja-build \
                   cmake libglib2.0-0 \
                   libopenmpi-dev \
                   libgl1-mesa-glx \
                   libsm6 libxext6 libxrender-dev ffmpeg \
                   gfortran libopenblas-dev liblapack-dev \
                   software-properties-common \
                   && add-apt-repository ppa:deadsnakes/ppa \
                   && apt-get update && apt-get install -y \
                   python3.10 python3.10-dev python3.10-distutils curl \
                   && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
                   && ln -sf python3.10 /usr/bin/python \
                   && ln -sf pip3 /usr/bin/pip \
    rm -rf /var/lib/apt/lists

RUN python -m pip install --no-cache-dir --upgrade pip setuptools && \
    python -m pip install --no-cache-dir mkl ninja

RUN python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PATH "$PATH:~/.local/bin"

WORKDIR /workspace

CMD ["/bin/bash"]
