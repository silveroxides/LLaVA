#!/bin/bash


# Store the current directory
previous_dir=$(pwd)

# Create conda environment and install packages
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
