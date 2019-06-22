FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update && apt install -y python3 \
    python3-pip

COPY requirements.txt /tmp/
WORKDIR /tmp

RUN pip3 install -r requirements.txt

WORKDIR /home/cnn-trading/src

CMD ["python3", "-c", "'from pipelines import *; train_pipeline_gram_binary()'"]




