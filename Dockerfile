FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update && apt install -y python3 \
    python3-pip \
    zip

COPY requirements.txt /tmp/

WORKDIR /tmp

RUN pip3 install -r requirements.txt
COPY kaggle.json /root/.kaggle/
RUN kaggle datasets download -d mczielinski/bitcoin-historical-data -p /home/cnn-trading/data

WORKDIR /home/cnn-trading/data
RUN unzip bitcoin-histrical-data.zip 

CMD ["/bin/bash"]



