# Using the latest TensorFlow beta available at the time and python version 3
# (GPU compatible)
FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

# Find this directory copied to the container in ~/mwp folder
ADD . mwp/

# Might add BERT later
# RUN pip install bert-tensorflow
# RUN pip install bert-embedding

# For the translation example's data
RUN pip install tensorflow-datasets

# Install the python dependencies we're using
RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn

# For equation evaluation
# RUN pip install sympy

