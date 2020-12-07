FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer "Mark"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake libcanberra-gtk-module \
    libopenblas-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 python3-pip python3-dev python3-tk python3-setuptools \
  && cd /usr/local/bin

# Science libraries and other common packages
RUN pip3 --no-cache-dir install --upgrade \
    pip setuptools

RUN pip3 --no-cache-dir install --upgrade \
    numpy scipy pygame matplotlib pyflann3 requests \
    tqdm argparse sklearn h5py seaborn pandas 

RUN pip3 --no-cache-dir install --upgrade \
    networkx dgl-cu100 pygsp pyntcloud \
    vg pyyaml 
#open3d ImportError: libGL.so.1: cannot open shared object file: No such file or directory

RUN pip3 install torch==1.1.0 torchvision==0.3.0
RUN pip3 install tensorflow-gpu==2.0

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /home
