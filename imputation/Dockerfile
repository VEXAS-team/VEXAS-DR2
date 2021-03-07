FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3-pip

RUN mkdir /imputer
WORKDIR /imputer

RUN python3 --version
RUN pip3 install --upgrade pip && pip3 --version
COPY requirements.txt /imputer
RUN pip3 install -r requirements.txt

ADD . /imputer/

RUN python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
RUN python3 -c "import keras; print(keras.__version__)"
