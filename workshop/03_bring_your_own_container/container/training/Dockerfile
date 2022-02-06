FROM tensorflow/tensorflow:2.4.1-gpu

# install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# copy the training script inside the container
COPY train.py /opt/ml/code/train.py

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train.py