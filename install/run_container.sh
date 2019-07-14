#!/usr/bin/env bash

WORKDIR="/home/development/"
SHARE_DIR=${WORKDIR}/go/src/github.com/goMLLibrary
INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${INSTALL_SCRIPT_DIR}/.." && pwd)"
DOCKER_CONTAINER_NAME="go_ml_container"
DOCKER_IMG_NAME="go_ml"

# build docker image
docker build -t ${DOCKER_IMG_NAME} -f ${INSTALL_SCRIPT_DIR}/Dockerfile --build-arg WORKDIR=${WORKDIR} .

# build docker container
docker run -it \
    -v ${ROOT_DIR}:${SHARE_DIR} \
    --name ${DOCKER_CONTAINER_NAME} \
    ${DOCKER_IMG_NAME} \
    /bin/bash
