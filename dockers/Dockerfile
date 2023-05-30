FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="Fuliucansheng"
LABEL repository="unitorch"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1
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
                   cmake \
                   libopenmpi-dev \
                   libgl1-mesa-glx \
                   gfortran libopenblas-dev liblapack-dev \
                   python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    python3 -m pip install --no-cache-dir mkl \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 ninja

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PATH "$PATH:~/.local/bin"

WORKDIR /workspace

CMD ["/bin/bash"]
