docker rm gan

docker run -it \
    --gpus all \
    -p 8888:8888 \
    -v /home/michal/Projects/gan/data:/gan/data \
    -v /home/michal/Projects/gan/logs:/gan/logs \
    -v /home/michal/Projects/gan/notebooks:/gan/notebooks \
    -v /home/michal/Projects/gan/models:/gan/models \
    -v /home/michal/Projects/gan/gan:/gan/gan \
    --name gan \
    gan $*
