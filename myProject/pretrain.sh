#!/bin/bash

cd ..

deepspeed /kaggle/working/LLaVA/llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --bits 4 \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./dataset/chat.json \
    --image_folder ./dataset \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ./checkpoints/llava-vicuna-v5-3-7b \
    --max_steps 50 \
    --num_experts_per_tok 2 \
    --num_experts 4 \
    --aux_loss_coef 0.01 \
    --mm_projector_type 'sparse_moe'\
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "train_xformers"