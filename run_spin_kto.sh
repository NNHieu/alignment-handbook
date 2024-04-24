#!/bin/bash
set -e

# Iteration 0
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_0.yaml
ACCELERATE_LOG_LEVEL=info conda run -n handbook accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_0.yaml

# Iteration 1
conda run -n vllm python generate_spin_data.py --data_frac 1 --frac_len 5001 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-0-full';
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_1.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_1.yaml

# Iteration 2
conda run -n vllm python generate_spin_data.py --data_frac 1 --frac_len 5001 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-1-full';
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_2.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_2.yaml

# Iteration 3
conda run -n vllm python generate_spin_data.py --data_frac 1 --frac_len 5001 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-2-full';
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_3.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_3.yaml
