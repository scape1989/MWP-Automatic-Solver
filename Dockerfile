FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

WORKDIR /home/translator

RUN pip install --upgrade pip
RUN pip install --upgrade virtualenv
RUN virtualenv /home/translator
RUN . /home/translator/bin/activate
RUN pip install tensorflow-datasets
RUN pip install nltk