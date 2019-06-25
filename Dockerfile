FROM tensorflow/tensorflow:nightly-gpu
WORKDIR /home/mwp
ADD . /home/mwp

RUN chmod -R 777 /home/mwp
RUN mkdir /.mxnet
RUN chmod -R 777 /.mxnet

RUN apt-get install -y python3-pip
RUN apt-get install -y python3.7

RUN pip3 install bert-embedding
RUN pip3 install tensorflow-gpu
RUN pip3 install tensorflow-datasets
RUN pip3 install keras-transformer
RUN pip3 install Keras
RUN pip3 install numpy
