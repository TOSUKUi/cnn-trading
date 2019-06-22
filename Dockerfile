FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update && apt install -y python3 \
    python3-pip

COPY requirements.txt /tmp/
WORKDIR /tmp

RUN pip3 install -r reuqirements.txt

WORKDIR /home/cnn-trading

CMD ["/bin/bash"]




