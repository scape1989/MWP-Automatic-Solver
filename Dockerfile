FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

WORKDIR /home/mwp
ADD . /home/mwp

RUN pip install keras
RUN pip install tensorflow-datasets
RUN pip install numpy
RUN pip install nltk
