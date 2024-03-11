# Use the rocm/dev-ubuntu-22.04:latest image as the base
FROM rocm/dev-ubuntu-22.04:latest

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Optionally, set symbolic links for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip
