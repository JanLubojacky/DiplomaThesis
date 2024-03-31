# Use the rocm/dev-ubuntu-22.04:latest image as the base, this uses python 3.10
FROM rocm/dev-ubuntu-22.04:latest

RUN apt-get update && apt-get install -y \
    unzip

WORKDIR /diploma
COPY requirements.txt /diploma

RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7

# get pytorch geometric for rocm
# url from here https://github.com/Looong01/pyg-rocm-build/releases/tag/4
# has to match python version
RUN curl -LO https://github.com/Looong01/pyg-rocm-build/releases/download/4/torch-2.2-rocm-5.7-py310-linux_x86_64.zip \
    && unzip torch-2.2-rocm-5.7-py310-linux_x86_64.zip \
    && pip3 install ./torch-2.2-rocm-5.7-py310-linux_x86_64/* \
    && rm -rf torch-2.2-rocm-5.7-py310-linux_x86_64.zip

ENV JUPYTER_ENABLE_LAB=yes
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
EXPOSE 8888

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
