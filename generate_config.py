import yaml
from pathlib import Path

def generate_spin_config(template, base_model, generated_dataset_path, iteration, loss_type):
    with open(template, 'r') as f:
        a = yaml.safe_load(f)
    model_name = base_model.split("/")[-1]
    
    a['hub_model_id'] = f"{model_name}-spin-{loss_type}-iter{iteration}"
    a['output_dir'] = str(Path(base_model) / "spin" / loss_type / f"iter{iteration}")
    a['precompute_ref_log_probs_path'] = str(Path(a['output_dir']) / 'ref_logprobs.pkl')
    a['loss_type'] = loss_type

    if iteration > 0:
        prev_id = f"{model_name}-spin-{loss_type}-iter{iteration - 1}"
        a['model_name_or_path'] = str(Path(base_model) / "spin" / loss_type / f"iter{iteration - 1}")
        a['dataset_mixer'] = {
            str(Path(f'/home/ubuntu/hieu.nn/Lang/SPIN/data/spin_data/{prev_id}/ultrachat_200k_generated/{iteration - 1}_5000')): 1.0,
            str(Path(f'/home/ubuntu/hieu.nn/Lang/SPIN/data/spin_data/{a["hub_model_id"]}/ultrachat_200k_generated/{iteration}_5000')): 1.0
        }
    elif iteration == 0:
        a['model_name_or_path'] = str(Path(base_model))
        a['dataset_mixer'] = {
            generated_dataset_path: 1.0
        }
    return a

a = generate_spin_config('recipes/stablelm-2-1_6b/spin-dpo/template.yaml', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-sft-full', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/spin_data/0_5000', 
                         0, 
                         "sigmoid")
with open("recipes/stablelm-2-1_6b/spin-dpo/config_full_0.yaml", 'w') as f:
    yaml.safe_dump(a, f)

a = generate_spin_config('recipes/stablelm-2-1_6b/spin-dpo/template.yaml', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-sft-full', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/spin_data/0_5000', 
                         1, 
                         "sigmoid")
with open("recipes/stablelm-2-1_6b/spin-dpo/config_full_1.yaml", 'w') as f:
    yaml.safe_dump(a, f)

a = generate_spin_config('recipes/stablelm-2-1_6b/spin-dpo/template.yaml', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-sft-full', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/spin_data/0_5000', 
                         2, 
                         "sigmoid")
with open("recipes/stablelm-2-1_6b/spin-dpo/config_full_2.yaml", 'w') as f:
    yaml.safe_dump(a, f)

a = generate_spin_config('recipes/stablelm-2-1_6b/spin-dpo/template.yaml', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/stablelm-2-1_6b-sft-full', 
                         '/home/ubuntu/hieu.nn/Lang/alignment-handbook/data/spin_data/0_5000', 
                         3, 
                         "sigmoid")
with open("recipes/stablelm-2-1_6b/spin-dpo/config_full_3.yaml", 'w') as f:
    yaml.safe_dump(a, f)