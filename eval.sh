#!/bin/bash
set -e

export MODEL_PATH="nnheui/stablelm-2-1_6b-sft-full-spin-sigmoid-iter0_1_61101_extlabel"
export EVAL_PATH="eval_outputs/stablelm-2-1_6b-spin-dpo-61k-0-extlabel"
# export CUDA_VISIBLE_DEVICES=1,2
# export GPU_IDS='2,3,4,5'
export GPU_IDS='all'
# accelerate launch --gpu_ids $GPU_IDS -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples
# accelerate launch --gpu_ids $GPU_IDS -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples
# accelerate launch --gpu_ids $GPU_IDS -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples
accelerate launch --gpu_ids $GPU_IDS -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples
accelerate launch --gpu_ids $GPU_IDS -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples
accelerate launch --gpu_ids $GPU_IDS -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples
