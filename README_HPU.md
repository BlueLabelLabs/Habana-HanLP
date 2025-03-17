# HanLP with Intel® Gaudi® HPU

## Overview
HanLP is a multilingual Natural Language Processing (NLP) library built on PyTorch and TensorFlow 2.x, designed for researchers and enterprises. It supports state-of-the-art deep learning techniques and provides a wide range of NLP tasks, including:
- Tokenization
- Lemmatization
- Part-of-Speech (POS) Tagging
- Named Entity Recognition (NER)
- Dependency Parsing
- Semantic Role Labeling (SRL)
- Abstract Meaning Representation (AMR) Parsing

## Running HanLP on Intel® Gaudi® HPU
This guide provides steps to build and run HanLP using Intel® Gaudi® HPU for accelerated performance.

### 1. Build the Docker Image
To run HanLP on Intel® Gaudi® HPU, you need to build a Docker image with the required environment.

```bash
docker build -t hanlp_hpu:latest -f Dockerfile.hpu .
```

The `Dockerfile.hpu` uses the `vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest` base image, ensuring compatibility with Habana Gaudi accelerators.

### 2. Run the Docker Container
Once the image is built, launch the container with the following command:

```bash
docker run -it --runtime=habana hanlp_hpu:latest
```

### 3. Using HanLP Inside the Container
After starting the container, you can use HanLP through Python. Here’s an example of loading a pre-trained model and processing text:

```python
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
print(HanLP(['HanLP provides state-of-the-art NLP techniques.']))
```

This runs HanLP’s multi-task model, performing tokenization, POS tagging, dependency parsing, and more.

## Additional Resources
For more details, visit the [HanLP Documentation](https://hanlp.hankcs.com/docs/).

