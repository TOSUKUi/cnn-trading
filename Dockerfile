FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://linux.yz.yamagata-u.ac.jp/ubuntu/%g" /etc/apt/sources.list && apt update && apt install -y python3 \
    python3-pip \
    zip \
    nodejs \
    npm

RUN pip3 install pipenv

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

COPY requirements.txt  /tmp
WORKDIR /tmp/
RUN pip3 install -r requirements.txt
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter labextension install jupyterlab_tensorboard @lckr/jupyterlab_variableinspector  jupyterlab_vim  jupyterlab-nvdashboard @jupyter-widgets/jupyterlab-manager

WORKDIR /home/workspace
COPY kaggle.json /root/.kaggle/
RUN kaggle datasets download -d mczielinski/bitcoin-historical-data -p /home/workspace/data

CMD ["jupyter-lab", "--ip", "0.0.0.0", "--allow-root"]



