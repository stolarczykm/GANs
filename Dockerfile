FROM tensorflow/tensorflow:1.9.0-gpu-py3
ADD requirements.txt /
RUN add-apt-repository ppa:djcj/hybrid && \
    apt-get update && \
    apt-get install -y ffmpeg
RUN pip install -r /requirements.txt
ADD src /src
ADD notebooks /notebooks
WORKDIR /notebooks
CMD ["/run_jupyter.sh", "--allow-root"]
