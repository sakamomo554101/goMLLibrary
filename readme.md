# goMLLibrary

## Overview

goMLLibrary is the library of neural network components in go.

## Environment

* Go 1.12(use Go Modules)
* Docker

## Build&Run

* need to use Go Modules

```bash
$ export GO111MODULE=on
```

* run mnist sample

```bash
$ go run ./main/MnistSample.go
```
## Layer

### Activation

* Relu
* Sigmoid
* Tanh
* SoftmaxWithCrossEntropy

### NeraulNetworkCell

* Affine

### Optimizer

* SGD

## Docker

### Build Container

* create docker container and attach it

```bash
$ bash install/run_container.sh
```

* setup the environment of docker container

```bash
$ cd go/src/github.com/goMLLibrary
$ bash install/setup_container.sh
$ source ~/.bash_profile
```

## Run Sample TVM code

If you use sample tvm code, you need to create docker container.
Following command can be done in docker container.

### Move goMLLibrary root path

```bash
$ cd go/src/github.com/goMLLibrary
```

### Get onnx model

```bash
$ cd python
$ python3 tvm_wrapper.py
$ cd ../
```

### Build sample code

```bash
$ export GOPATH=/home/development/go
$ bash build_tvm_sample.sh
```

### Run Inference

```bash
$ go run tvm_sample/GotvmSample
```
