#!/bin/bash

# Create a folder
folder_name="dataset"
mkdir -p "$folder_name"
echo "Folder $folder_name created successfully."

# Change directory to the dataset folder
cd "$folder_name"

# Download files
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip

# Unzip the files
unzip -q images.zip

# Delete unnecessary zip file
rm -f images.zip
echo "File images.zip deleted successfully."
