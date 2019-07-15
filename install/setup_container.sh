#!/usr/bin/env bash

WORKDIR=/home/development

# setup environment
echo GOROOT=/usr/local/go >> ~/.bash_profile
echo GOPATH=$WORKDIR/go >> ~/.bash_profile
echo PATH=\$GOPATH/bin:\$GOROOT/bin:\$PATH >> ~/.bash_profile
echo export GO111MODULE=on >> ~/.bash_profile
source ~/.bash_profile

# build gotvm
cd $WORKDIR/tvm/golang
make

# move gotvm
cd $WORKDIR
GOML_PATH=$WORKDIR/go/src/github.com/goMLLibrary
GOTVM_PATH=$GOML_PATH/core/tvm_wrapper/gotvm.a
if [ -e $GOTVM_PATH ]; then
    rm -r $GOTVM_PATH
fi
mv $WORKDIR/tvm/golang/gopath/src/gotvm/gotvm.a $GOML_PATH/core/tvm_wrapper
