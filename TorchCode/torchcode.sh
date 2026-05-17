#!/usr/bin/env bash

NAME=torchcode
IMAGE=ghcr.io/duoan/torchcode:latest
CWD=$(pwd)
WORK_DIR="${CWD}/torchcode_work"

if [[ ! -d ${WORK_DIR} ]]; then
  mkdir "${WORK_DIR}"
fi

if ! docker inspect "${NAME}" >/dev/null 2>&1; then
  docker pull "${IMAGE}"
  docker run -d \
    --name "${NAME}" \
    --restart unless-stopped \
    -p 8888:8888 -e PORT=8888 \
    -v "${WORK_DIR}:/work" \
    "${IMAGE}"
else
  docker start "${NAME}"
fi

echo "Ready: http://localhost:8888"
echo "# 停止方法 : docker stop torchcode"
