#!/bin/bash

# Store the current directory
previous_dir=$(pwd)

# Go up one directory to the main directory
cd ..

# Create a folder
folder_name="dataset"
mkdir -p "$folder_name"
echo "Folder $folder_name created successfully."

# Change directory to the dataset folder
cd "$folder_name"

# Download files
wget https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/chat.json
wget https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip

# Unzip the files
unzip -q images.zip

# Delete unnecessary zip file
rm -f images.zip
echo "File images.zip deleted successfully."

# Go back to the previous directory
cd "$previous_dir"
