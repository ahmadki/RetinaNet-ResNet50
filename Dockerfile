ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.08-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace/ssd

# Install requirements
COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . .
