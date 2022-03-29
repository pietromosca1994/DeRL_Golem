#!/bin/bash

export DOCKER_VM_NAME=baselines3_vm

# start docker service
echo '[INFO] Start Docker Service'
sudo service docker start
# build docker image 
echo '[INFO] Build Docker VM image'
sudo docker build . -t $DOCKER_VM_NAME
# build golem compatible image ('.gvmi') from docker image and push it to the remote repository
echo '[INFO] Build Golem compatible VM image'
gvmkit-build $DOCKER_VM_NAME:latest 
# push golem compatible image ('.gvmi') to the remote repository
echo '[INFO] Pushing Golem compatible VM image to remote repository'
gvmkit-build $DOCKER_VM_NAME:latest --push