#!/usr/bin/env bash

PRJ_PATH=$GOPATH/src/github.com/goMLLibrary
TVM_WRAPPER_PATH=${PRJ_PATH}/core/tvm_wrapper
LIB_ROOT_PATH=${PRJ_PATH}/core/tvm_wrapper
SAMPLE_PATH=${PRJ_PATH}/tvm_sample/

# clean wrapper module if it exist
make clean -C tvm_sample

# build
go tool compile -D $TVM_WRAPPER_PATH -o $SAMPLE_PATH/tvm_wrapper.a -p tvm_wrapper -pack $LIB_ROOT_PATH/TvmWrapper.go

# make
make -C tvm_sample
