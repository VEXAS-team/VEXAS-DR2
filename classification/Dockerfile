FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3-pip

RUN mkdir /code
WORKDIR /code

RUN pip3 install --upgrade pip && pip3 --version
COPY requirements.txt /code
RUN pip3 install -r requirements.txt

ADD . /code/
RUN pip install shap==0.37.0
RUN pip install jupyter notebook
