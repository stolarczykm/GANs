docker run -it \
    --runtime=nvidia \
    -p 8888:8888 \
    -v /home/michal/Projects/gan/data:/data \
    -v /home/michal/Projects/gan/logs:/logs \
    -v /home/michal/Projects/gan/notebooks:/notebooks \
    -v /home/michal/Projects/gan/src:/src \
    --name gan \
    gan
