#!/bin/bash
set -e

# conda run -n vllm python generate_spin_data.py --data_frac 1 --frac_len 61101 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-sft-full';
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-dpo/config_full_0_2.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-dpo/config_full_0_61k_subset.yaml
