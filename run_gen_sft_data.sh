# CUDA_VISIBLE_DEVICES=3 python generate_sft_data.py  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model_alias tinyllama-1_1b --data_frac 0 --frac_len 50000
CUDA_VISIBLE_DEVICES=3 python generate_sft_data.py  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model_alias tinyllama-1_1b --data_frac 1 --frac_len 50000
CUDA_VISIBLE_DEVICES=3 python generate_sft_data.py  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model_alias tinyllama-1_1b --data_frac 2 --frac_len 50000
CUDA_VISIBLE_DEVICES=3 python generate_sft_data.py  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model_alias tinyllama-1_1b --data_frac 3 --frac_len 50000
CUDA_VISIBLE_DEVICES=3 python generate_sft_data.py  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model_alias tinyllama-1_1b --data_frac 4 --frac_len 50000
