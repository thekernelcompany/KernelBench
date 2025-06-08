# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    cmake \
    vim \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Triton
RUN pip install triton

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project
COPY . .

# Install the project in development mode
RUN pip install -e .

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV TRITON_PRINT_AUTOTUNING=1

# Default command
CMD ["/bin/bash"] 