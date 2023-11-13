#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1
# #SBATCH --constraint=v100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-025-xnli-lm_eval_pilot_bloom_fish_de_560m

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-025-xnli/lm_eval_pilot_bloom_fish_de_560m.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-025-xnli/lm_eval_pilot_bloom_fish_de_560m.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.9.0
module load gitlfs/2.7.1
source $FP_BIGS/env_lm_eval_harness_old/bin/activate

python3 /users/zyong2/data/zyong2/bigscience/data/external/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer="bigscience/bloom-560m",pretrained="/users/zyong2/data/zyong2/bigscience/data/processed/024/bloom-560m_de_fish_100000samples_-1vocab_original-frozen" \
--tasks xnli_de