#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 01:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 
#SBATCH --cpus-per-task=32
#SBATCH --output=log/slurm/evaluation/scienceQA/log-%A-%a.log 
#SBATCH -J SQA

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

export CUDA_VISIBLE_DEVICES=0


module purge
module load apptainer

model_paths=('ckpts_it/baseline/base-v1.5-13b-finetune' 'ckpts_it/baseline/base-v1.5-13b-finetune-v2' 'ckpts_it/moe_full/moe-e4t2-finetune' 'ckpts_it/moe_full/moe-e5t3-finetune' 'ckpts_it/moe/moe-e8t2-finetune')
file_names=('base-v2' 'base-v2' 'moe-e4t4' 'moe-e5t3' 'moe-e8t2')
outputs=('baseline_output' 'moe-e4t4_output' 'moe-e5t3_output' 'moe-e8t2_output')
results=('baseline_result' 'moe-e4t4_result' 'moe-e5t3_result' 'moe-e8t2_result')

current_file_name=${file_names[${SLURM_ARRAY_TASK_ID}]}
current_output=${outputs[${SLURM_ARRAY_TASK_ID}]}
current_result=${results[${SLURM_ARRAY_TASK_ID}]}
current_model_path=${model_paths[${SLURM_ARRAY_TASK_ID}]}

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava.sif \
python3 -m llava.eval.model_vqa_science \
    --model-path ${current_model_path} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_file_name}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava.sif \
python3 llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_file_name}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_output}.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_result}.json
