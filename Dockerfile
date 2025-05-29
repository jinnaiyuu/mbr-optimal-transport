# FROM nvcr.io/nvidia/pytorch:21.07-py3
# FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:23.12-py3

LABEL MAINTAINER=yuu

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /code/mbr


RUN apt-get update && \
    apt-get install -y python3-pip git default-jre cpanminus && \
    pip3 install six && \
    cpanm --force XML::Parser && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code/mbr

COPY requirements.txt /code/mbr/requirements.txt
RUN TMPDIR=/var/tmp pip3 install -r /code/mbr/requirements.txt && apt-get update && apt-get install -y jq curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U git+https://github.com/pltrdy/pyrouge && \
    pip3 install -U git+https://github.com/lucadiliello/bleurt-pytorch.git && \
    pip3 install -U git+https://github.com/neural-dialogue-metrics/Distinct-N.git

COPY mbr /code/mbr/mbr

COPY experiments /code/mbr/experiments
COPY prompts /code/mbr/prompts

RUN mkdir -p /code/mbr/results

ENTRYPOINT ["/bin/bash"]