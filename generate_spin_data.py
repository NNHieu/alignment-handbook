import os
from datasets import load_dataset
import argparse
import json
from pathlib import Path
import torch
import pyarrow.parquet as pq
import logging
import os
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import datasets
import pandas as pd
import gc
import os
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,4'
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    # parser.add_argument('--output_dir', type=str, default='generated/iter1')
    # parser.add_argument('--world_size', type=int, default=8) # controls the number of gpus vLLM is allowed to use
    # parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    # parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()
args = parse_arguments()

data_path = 'HuggingFaceH4/ultrachat_200k'
output_dir = Path('data/spin_data')
data_frac = args.data_frac
# frac_len = 5001
frac_len = args.frac_len


# model_path = "/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-spin-dpo-0-full"
model_path = args.model
model_alias = args.model.split('/')[-1]

output_dir = output_dir / model_alias
output_dir.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(data_path, split='train_sft')

def apply_chat_template(example, tokenizer, ):
    messages = example['messages']
    if example["messages"][0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True,)
    example['prompt_text'] = prompt
    return example

appliedtemplate_datasets = dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        # "task": "sft",
        # "auto_insert_empty_system_msg": True,
    },
    num_proc=12,
    # remove_columns=column_names,
    desc="Applying chat template",
)

appliedtemplate_datasets = appliedtemplate_datasets.shuffle(42)

if frac_len > 0:
    sub_len = frac_len 
    if sub_len*(data_frac+1) > len(appliedtemplate_datasets):
        appliedtemplate_datasets = appliedtemplate_datasets.select(range(sub_len*data_frac, len(appliedtemplate_datasets)))
    else:
        appliedtemplate_datasets = appliedtemplate_datasets.select(range(sub_len*data_frac,sub_len*(data_frac+1)))
# else:
#     appliedtemplate_datasets = appliedtemplate_datasets[:]

llm = LLM(
    model=model_path,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.7, 
)
sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)
results_gathered = list(map(lambda x: x.outputs[0].text, 
                                  llm.generate(appliedtemplate_datasets['prompt_text'], sampling_params)))
results = [r.replace("</s>","").lstrip() for r in results_gathered]

generated_dataset =[]
for idx in range(len(results)):
    d = {
        "prompt": appliedtemplate_datasets[idx]['prompt'],
        "prompt_id": appliedtemplate_datasets[idx]['prompt_id'],
        "chosen": appliedtemplate_datasets[idx]['messages'][1:3],
        "rejected": [appliedtemplate_datasets[idx]['messages'][1],
                     {"role": "assistant", "content": results[idx]}],
        }
    generated_dataset.append(d)

ds = datasets.Dataset.from_pandas(pd.DataFrame(data=generated_dataset))
ds = ds.train_test_split(test_size=1)
ds['train'].to_parquet(output_dir / f"ultrachat_200k_generated/{data_frac}_{frac_len}/train/data.parquet")
ds['test'].to_parquet(output_dir / f"ultrachat_200k_generated/{data_frac}_{frac_len}/test/data.parquet")

# Delete the llm object and free the memory
destroy_model_parallel()
del llm
gc.collect()
torch.cuda.empty_cache()
torch.distributed.destroy_process_group()
print("Successfully delete the llm pipeline and free the GPU memory!")