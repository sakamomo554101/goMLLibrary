# base is Ubuntu 16.04
FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive
MAINTAINER Dummy <dummy@dummy.com>

# update apt-get
RUN set -x && \
    apt-get update --fix-missing && \
    apt-get -y upgrade

# install git
RUN apt-get install -y git

# setup workdir
ARG WORKDIR="/home/development/"
WORKDIR $WORKDIR

# create golang folder
RUN set -x && \
    mkdir -p go/pkg go/bin go/src && \
    mkdir -p go/src/github.com/goMLLibrary

# Install Standard Command
RUN set -x && \
    apt-get install -y curl && \
    apt-get install -y vim && \
    apt-get install -y software-properties-common

# clone tvm repository
RUN set -x && \
    git clone --recursive https://github.com/dmlc/tvm

# install python packages to need to build tvm and so on.
RUN set -x && \
    bash tvm/docker/install/ubuntu_install_core.sh && \
    bash tvm/docker/install/ubuntu_install_python.sh && \
    bash tvm/docker/install/ubuntu_install_python_package.sh && \
    bash tvm/docker/install/ubuntu_install_llvm.sh && \
    bash tvm/docker/install/ubuntu_install_redis.sh && \
    bash tvm/docker/install/ubuntu_install_java.sh && \
    bash tvm/docker/install/ubuntu_install_antlr.sh && \
    bash tvm/docker/install/ubuntu_install_nnpack.sh && \
    bash tvm/docker/install/ubuntu_install_onnx.sh

# Set link for add-apt-repository command
RUN set -x && \
    apt-get remove -y python3-apt && \
    apt-get install -y python3-apt

# Install golang 1.12
RUN set -x && \
    wget https://dl.google.com/go/go1.12.6.linux-amd64.tar.gz && \
    tar -xvf go1.12.6.linux-amd64.tar.gz && \
    mv go /usr/local && \
    rm go1.12.6.linux-amd64.tar.gz

# Install go packages
RUN set -x && \
    apt-get install -y golint

# build nnvm/tvm module
RUN set -x && \
    cd tvm && \
    mkdir build && \
    cp cmake/config.cmake build && \
    cd build && \
    echo set\(USE_LLVM llvm-config-6.0\) >> config.cmake && \
    echo set\(USE_NNPACK ON\) >> config.cmake && \
    echo set\(NNPACK_PATH /home/development/NNPACK/build\) >> config.cmake && \
    echo set\(USE_ANTLR ON\) >> config.cmake && \
    echo set\(USE_GRAPH_RUNTIME_DEBUG ON\) >> config.cmake && \
    cmake .. && \
    make -j4

# setup python environment for nnvm/tvm
RUN set -x && \
    cd tvm/python && \
    python3 setup.py install --user && \
    cd .. && \
    cd topi/python && \
    python3 setup.py install --user && \
    cd ../.. && \
    cd nnvm/python && \
    python3 setup.py install --user
