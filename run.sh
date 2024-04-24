# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_sft.py recipes/stablelm-2-1_6b/sft/config_full.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/dpo/config_full.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num-processes=6 scripts/run_preref_dpo.py recipes/stablelm-2-1_6b/dpo/config_full.yaml

export MODEL_PATH="/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-dpo-full"
export EVAL_PATH="eval_outputs/stablelm-2-1_6b-dpo-full"
# export CUDA_VISIBLE_DEVICES=1,2
accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained=${MODEL_PATH},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples
