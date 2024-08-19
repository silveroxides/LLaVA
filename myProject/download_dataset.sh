#!/bin/bash

# Create a folder
folder_name="dataset"
mkdir -p "$folder_name"
echo "Folder $folder_name created successfully."

# Change directory to the dataset folder
cd "$folder_name"

wget https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/chat.json
wget https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip


# Download files
# wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json

# # Go to the previous directory
# cd ..

# # Create a folder named images
# mkdir -p images

# # Change directory to the images folder
# cd images

# wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
# Unzip the files
unzip -q images.zip

# Delete unnecessary zip file
rm -f images.zip

echo "File images.zip deleted successfully."
