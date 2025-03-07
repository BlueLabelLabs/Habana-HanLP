# HanLP Intel® Gaudi® HPU Usage

This document describes the setup and deployment of HanLP with Intel® Gaudi® HPU environments.

## Intel® Gaudi® HPU Usage

### Build the Docker Image

To use Intel® Gaudi® HPU for running HanLP, start by building a Docker image with the appropriate environment setup.

```bash
docker build -t hanlp_hpu:latest -f Dockerfile.hpu .
```

In the `Dockerfile.hpu`, we use the `vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest` base image. Ensure the base image version matches your setup.

See the [PyTorch Docker Images for the Intel® Gaudi® Accelerator](https://developer.habana.ai/catalog/pytorch-container/) for more information.

### Run the Container

Start the container using the following command:

```bash
docker run -it --runtime=habana hanlp_hpu:latest
```

### Dockerfile (Dockerfile.hpu)

```Dockerfile
# Use the official Gaudi Docker image with PyTorch
FROM vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest

# Set environment variables for Habana
ENV HABANA_VISIBLE_DEVICES=all
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV PT_HPU_LAZY_ACC_PAR_MODE=0
ENV PT_HPU_ENABLE_LAZY_COLLECTIVES=1

# Set timezone to UTC and install essential packages
ENV DEBIAN_FRONTEND="noninteractive" TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    tzdata \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone HanLP repository
RUN git clone https://github.com/hankcs/HanLP.git /workspace/hanlp

WORKDIR /workspace/hanlp

# Copy HPU-specific requirements
COPY requirements_hpu.txt /workspace/requirements_hpu.txt

# Install Python packages
RUN pip install --upgrade pip \
    && pip install -e ".[full]" \
    && pip install -r /workspace/requirements_hpu.txt
```

### requirements_hpu.txt

```text
optimum-habana==1.14.1
transformers==4.45.2
huggingface-hub==0.26.2
tiktoken==0.8.0
torch-geometric==2.6.1
numba==0.60.0
```
