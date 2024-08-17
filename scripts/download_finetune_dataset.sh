# move to target dir
# Create a folder
folder_name="./playground/data"
cd "$folder_name"

# json
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json

# coco
mkdir -p coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip
rm -f train2017.zip
cd ..

#gqa
mkdir -p gqa
cd gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip -q images.zip
rm -f images.zip
cd ..

#text_vqa
mkdir -p textvqa
cd textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip -q train_val_images.zip
rm -f train_val_images.zip
cd ..

#vq
mkdir -p vg
cd vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# Unzip the first archive into the VG_100K directory
unzip -q images.zip
rm -f images.zip

# Unzip the second archive into the VG_100K_2 directory
unzip -q images2.zip
rm -f images2.zip
