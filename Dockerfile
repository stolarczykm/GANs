FROM tensorflow/tensorflow:2.0.0-gpu-py3
RUN python -V
ADD requirements.txt /
RUN add-apt-repository ppa:djcj/hybrid && \
    apt-get update && \
    apt-get install -y ffmpeg
RUN pip install -r /requirements.txt
ADD . /gan
WORKDIR /gan
RUN pip install -e . 
CMD ["/run_jupyter.sh", "--allow-root"]
