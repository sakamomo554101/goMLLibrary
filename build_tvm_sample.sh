#!/usr/bin/env bash

PRJ_PATH=$GOPATH/src/github.com/goMLLibrary
TVM_WRAPPER_PATH=${PRJ_PATH}/core/tvm_wrapper
LIB_PATH=${PRJ_PATH}/tvm_sample/tvm_wrapper.a

# clean wrapper module if it exist
make clean -C tvm_sample

# build
go tool compile -D $TVM_WRAPPER_PATH -o $LIB_PATH -p tvm_wrapper -pack core/tvm_wrapper/TvmWrapper.go

# make
make -C tvm_sample
