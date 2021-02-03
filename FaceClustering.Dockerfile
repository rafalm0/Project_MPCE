# FROM ubuntu:latest
# RUN apt-get update && apt-get -y update
# RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
# RUN pip3 -q install pip --upgrade
from python:3.8

# ENV DEBIAN_FRONTEND=noninteractive 
RUN pip3 install matplotlib
RUN apt update 
RUN apt install -y python3-scipy
RUN pip3 install ipython 
RUN apt-get install -y python3-dev python3-setuptools
RUN apt-get install -y python3-dev python3-setuptools
RUN apt-get install -y python3-setuptools
RUN echo "####### PYNPUT ########"
RUN pip3 install pynput
RUN echo "######## DLIB #########"
RUN apt-get update
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libopenblas-dev liblapack-dev libatlas-base-dev
RUN apt-get install -y libx11-dev libgtk-3-dev
RUN pip3 install numpy
RUN pip3 install dlib
RUN apt-get install -y build-essential cmake pkg-config
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -y libxvidcore-dev libx264-dev
RUN apt-get install -y libgtk2.0-dev libgtk-3-dev
RUN apt-get install -y libatlas-base-dev gfortran
RUN apt-get install -y python3 python3-setuptools python3-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

RUN pip3 install numpy
RUN pip3 install face_recognition

RUN pip3 install opencv-python
RUN pip3 install pandas
RUN pip3 install imutils
RUN pip3 install -U scikit-learn
RUN pip3 install networkx
RUN apt-get install git
RUN pip3 install joblib
RUN pip3 install tqdm
RUN pip3 install jupyter

RUN git   clone https://github.com/samuel1sm/Project_MPCE 

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# CMD ["sh","user/Start.sh"]z
ENTRYPOINT chmod +x /Project_MPCE/user/Start.sh & sh /Project_MPCE/user/Start.sh
WORKDIR /Project_MPCE

# CMD ["bash /Project_MPCE/user/Start.sh"]
