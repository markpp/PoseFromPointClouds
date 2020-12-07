#!/bin/bash
xhost +local:
nvidia-docker run -it --rm --net=host \
  -e QT_GRAPHICSSYSTEM=native \
  -e CONTAINER_NAME=dockergnn-dev \
  --workdir=/home/code \
  -e DISPLAY=unix$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$(pwd)"/../:/home/code \
  dockergnn:latest
