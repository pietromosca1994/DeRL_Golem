FROM ubuntu:latest

RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends

RUN apt-get install cmake -y
RUN apt-get install curl -y
RUN apt-get install python3-pip -y
RUN pip install stable-baselines3[extra]

WORKDIR /golem

VOLUME /golem/input /golem/output /golem/src
