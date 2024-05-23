#!/bin/bash
set -e

# Iteration 0
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_0.yaml
# ACCELERATE_LOG_LEVEL=info conda run -n handbook accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_0.yaml

# Iteration 1
# conda run -n vllm python generate_spin_data.py --data_frac 1 --frac_len 5001 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-0-full';
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_1.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_1.yaml

# Iteration 2
# conda run -n vllm python generate_spin_data.py --data_frac 2 --frac_len 5001 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-1-full';
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_2.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_2.yaml

# Iteration 3
# conda run -n vllm python generate_spin_data.py --data_frac 3 --frac_len 5001 --model '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-2-full';
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=1 scripts/run_generate_reflogprobs.py recipes/stablelm-2-1_6b/spin-kto/config_full_3.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/spin-kto/config_full_3.yaml


# export MODEL_PATH="/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-0-full"
# export EVAL_PATH="eval_outputs/stablelm-2-1_6b-spin-kto-0-full"
# # export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples

# export MODEL_PATH="/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-1-full"
# export EVAL_PATH="eval_outputs/stablelm-2-1_6b-spin-kto-1-full"
# # export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples

# export MODEL_PATH="/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-2-full"
# export EVAL_PATH="eval_outputs/stablelm-2-1_6b-spin-kto-2-full"
# # export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples

# export MODEL_PATH="/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-kto-3-full"
# export EVAL_PATH="eval_outputs/stablelm-2-1_6b-spin-kto-3-full"
# # export CUDA_VISIBLE_DEVICES=1,2
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples
# accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples
