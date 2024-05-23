

python cal_logprobs.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --model_alias zephyr-7b-sft-full-SPIN-iter1 --data_frac 0 --frac_len 1000 --data_path UCLA-AGI/SPIN_iter1 --data_alias SPIN_iter1
python cal_logprobs.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter1 --model_alias zephyr-7b-sft-full-SPIN-iter1 --data_frac 0 --frac_len 1000 --data_path UCLA-AGI/SPIN_iter1 --data_alias SPIN_iter1

python cal_logprobs.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter1 --model_alias zephyr-7b-sft-full-SPIN-iter1 --data_frac 0 --frac_len 1000 --data_path UCLA-AGI/SPIN_iter2 --data_alias SPIN_iter2
python cal_logprobs.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter2 --model_alias zephyr-7b-sft-full-SPIN-iter2 --data_frac 0 --frac_len 1000 --data_path UCLA-AGI/SPIN_iter2 --data_alias SPIN_iter2

python cal_logprobs.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter2 --model_alias zephyr-7b-sft-full-SPIN-iter2 --data_frac 0 --frac_len 1000 --data_path UCLA-AGI/SPIN_iter3 --data_alias SPIN_iter3
python cal_logprobs.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter3 --model_alias zephyr-7b-sft-full-SPIN-iter3 --data_frac 0 --frac_len 1000 --data_path UCLA-AGI/SPIN_iter3 --data_alias SPIN_iter3


