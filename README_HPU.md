# Running HanLP on Intel Gaudi HPU

## Overview
HanLP is a multilingual Natural Language Processing (NLP) library built on PyTorch and TensorFlow 2.x, designed for researchers and enterprises. It supports state-of-the-art deep learning techniques and provides a wide range of NLP tasks, including:
- Tokenization
- Lemmatization
- Part-of-Speech (POS) Tagging
- Named Entity Recognition (NER)
- Dependency Parsing
- Semantic Role Labeling (SRL)
- Abstract Meaning Representation (AMR) Parsing

## Requirements

- Intel Gaudi HPU device
- Habana SynapseAI SDK installed
- HanLP library


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


### 3. Usage

HanLP supports running on Intel Gaudi HPU devices. You can specify the device when loading models:

```python
import hanlp

# Load model onto HPU device
tokenizer = hanlp.load('MODEL_NAME', devices='hpu')

# For multi-HPU usage, specify device indices
parser = hanlp.load('MODEL_NAME', devices=[0, 1]) # Uses HPU devices 0 and 1
```

## Example

Here's a complete example of using HanLP with HPU:

```python
import hanlp

# Load a tokenizer model on HPU
tokenizer = hanlp.load('MODEL_NAME', devices='hpu')

# Process some text
text = "你好，世界！" 
result = tokenizer(text)
print(result)
```

## Notes

- Make sure you have the Habana SynapseAI SDK properly installed and configured
- HPU support requires PyTorch with Habana extensions
- Not all models may be optimized for HPU - check model documentation for compatibility
- For best performance, batch processing is recommended when using HPU devices

## Additional Resources
For more details, visit the [HanLP Documentation](https://hanlp.hankcs.com/docs/).
