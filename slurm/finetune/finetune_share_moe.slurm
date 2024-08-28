#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 30:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:8
#SBATCH -N 1 
#SBATCH --cpus-per-task=64
#SBATCH --output=log/slurm/it_share_moe/log-%A-%a.log 
#SBATCH -J instruct_tune

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

model_name=('lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-13b-v1.5')
vision_tower=('openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336')
num_experts=(5 4 4 5 6)
num_experts_per_tok=(3 2 3 2 3)
num_layers=(2 2 2 2 2)
num_heads=(2 2 2 2 2)
batch_size=(16 16 16 16 16)
lr=(2e-5 2e-5 1e-3 1e-3)
tune_embed_tokens=(False False False False)
mm_projector_type=('sparse_moe' 'sparse_moe' 'sparse_moe' 'sparse_moe')
output_dir=('./ckpts_it/share_moe/llava-share_moe-e5t3' './ckpts/moe/llava-v1.5-13b-pretrain-v3' './ckpts/embed_v2/llava-v1.5-13b-pretrain-v3' './ckpts/embed_v2/llava-v1.5-13b-pretrain-v4' './ckpts/embed/llava-v1.5-13b-pretrain-v5')
PORT=$((29500 + SLURM_JOB_ID % 1000))


apptainer exec --nv /scratch/mi8uu/mrm/llava.sif deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path ${model_name[${SLURM_ARRAY_TASK_ID}]} \
--version v1 \
--data_path ./playground/data/llava_v1_5_mix665k.json \
--image_folder ./playground/data \
--vision_tower openai/clip-vit-large-patch14-336 \
--pretrain_mm_mlp_adapter ./ckpts/ref_embed/llava-v1.5-13b-pretrain-v1/mm_projector.bin \
--tune_embed_tokens ${tune_embed_tokens[${SLURM_ARRAY_TASK_ID}]} \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ${output_dir[${SLURM_ARRAY_TASK_ID}]} \
--num_experts_per_tok ${num_experts_per_tok[${SLURM_ARRAY_TASK_ID}]} \
--num_experts ${num_experts[${SLURM_ARRAY_TASK_ID}]} \
--num_layers ${num_layers[${SLURM_ARRAY_TASK_ID}]} \
--num_heads ${num_heads[${SLURM_ARRAY_TASK_ID}]} \
--aux_loss_coef 0.01 \
--share_moe True \
--cross_attention False \
--mm_projector_type ${mm_projector_type[${SLURM_ARRAY_TASK_ID}]} \
--num_train_epochs 1 \
--per_device_train_batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb \
--run_name "UVA-Pretrain-mlp"