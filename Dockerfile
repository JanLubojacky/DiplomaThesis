FROM rocm/dev-ubuntu-22.04:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev

# install packages using pip
