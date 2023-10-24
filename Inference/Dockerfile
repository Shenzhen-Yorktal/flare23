FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3-pip python3-dev vim \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && apt-get clean \
  && pip install --upgrade pip -i https://mirrors4.tuna.tsinghua.edu.cn/pypi/web/simple/

ADD models /models/
ADD nnUNet/nnUNet_double_fine /nnUNet/
ADD predict.sh ./

RUN mkdir -p /workspace/inputs && mkdir -p /workspace/outputs
RUN pip install pip -U -i https://mirrors4.tuna.tsinghua.edu.cn/pypi/web/simple/
RUN pip config set global.index-url https://mirrors4.tuna.tsinghua.edu.cn/pypi/web/simple/
RUN pip install -e /nnUNet/

