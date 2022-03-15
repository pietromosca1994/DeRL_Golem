#!/bin/bash
ENV_NAME=stable_baselines3

apt-get update && \
    apt-get upgrade -y --no-install-recommends
apt-get install cmake -y
apt-get install curl -y

conda create --name $ENV_NAME --clone base
conda activate $ENV_NAME
conda install pip
 ~/miniconda3/envs/$ENV_NAME/bin/pip install stable-baselines3[extra]