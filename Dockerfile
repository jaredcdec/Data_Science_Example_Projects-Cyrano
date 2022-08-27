FROM ubuntu:18.04

MAINTAINER Loreto Parisi loretoparisi@gmail.com

########################################  BASE SYSTEM
# set noninteractive installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    tzdata \
    curl

######################################## PYTHON3
#RUN apt-get install -y \
#    python3 \
#    python3-pip
#
RUN apt-get install -y python3
#RUN apt install pip
RUN apt-get remove python-pip python3-pip
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip3 uninstall transformers

# set local timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# transfer-learning-conv-ai
ENV PYTHONPATH /usr/local/lib/python3.6 
COPY . ./
COPY requirements.txt /tmp/requirements.txt
#####

#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3 get-pip.py
#RUN apt install python3-pip
RUN pip3 install numpy
RUN pip3 install setuptools
RUN pip3 install ez_setup
RUN pip3 install pandas
RUN pip3 install nltk
RUN pip3 install statistics
RUN pip3 install tensorflow
RUN pip3 install tensorflow_hub
RUN pip3 install transformers
RUN pip3 install --upgrade transformers
RUN pip3 install tabulate
RUN pip3 install operator
#RUN pip3 install re
RUN pip3 list
RUN python3 get_resources.py
#####
#RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# model zoo
RUN mkdir models && \
    curl https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz > models/finetuned_chatbot_gpt.tar.gz && \
    cd models/ && \
    tar -xvzf finetuned_chatbot_gpt.tar.gz && \
    rm finetuned_chatbot_gpt.tar.gz
    
CMD ["bash"]
