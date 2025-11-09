#base cude:11.8
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# install python3.10
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt update && \
    apt upgrade -y && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt install -y tzdata &&\
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && apt install python3.10 python3.10-distutils python3.10-dev python3.10-tk -y && \
    apt install curl gcc git libglib2.0-dev libsm6 libxrender1 libxext6 libgl1 -y

RUN curl https://bootstrap.pypa.io/get-pip.py -o /root/get-pip.py && \
    python3.10 /root/get-pip.py && rm -rf /root/get-pip.py && \
    python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip && \
    python3.10 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/&& \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# install requirements
RUN pip install -r docker_env/requirements.txt

CMD ["/bin/bash"]
