# colored comparison of the original and edited model outputs

import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored

ORIGINAL_MODEL_NAME = "PleIAs/Baguettotron"
EDITED_MODEL_DIR = "../edited_models/baguettotron-edited-60-71-k60"

ORIGINAL_MEMORIZED_FILE = "../data/unique_memorizations.jsonl"

SAMPLES_TO_SHOW = 5
SUFFIX_LEN = 48

def get_colored_diff(gt_text, gen_text):
    if gt_text == gen_text:
        return colored(gen_text, 'green')
    else:
        return colored(gen_text, 'red')

original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
original_model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_NAME, dtype=torch.bfloat16, device_map="auto").eval()

edited_tokenizer = AutoTokenizer.from_pretrained(EDITED_MODEL_DIR)
edited_model = AutoModelForCausalLM.from_pretrained(EDITED_MODEL_DIR, dtype=torch.bfloat16, device_map="auto").eval()

for tok in [original_tokenizer, edited_tokenizer]:
    tok.padding_side = 'left'; tok.pad_token = "[PAD]"
    
datasets_to_test = {"Original (Seen) Memorizations": ORIGINAL_MEMORIZED_FILE}

with torch.no_grad():
    for name, filepath in datasets_to_test.items():
        print(f"sampling from: {name} ({filepath})")

        with open(filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        
        sampled_data = random.sample(data, SAMPLES_TO_SHOW)

        for i, item in enumerate(sampled_data):
            prefix_ids = item['prefix_ids']
            gt_suffix_ids = item['suffix_ids']

            prefix_text = original_tokenizer.decode(prefix_ids)
            gt_suffix_text = original_tokenizer.decode(gt_suffix_ids)
            
            print(f"--- Example #{i+1} ---")
            print(colored("Prefix:", 'cyan'))
            print(f"...{prefix_text[-150:]}") # only print last 150 chars of prefix
            print("-" * 20)
            print(colored("Ground Truth Suffix:", 'cyan'))
            print(gt_suffix_text)
            print("-" * 20)

            # generate from original model
            inputs_original = original_tokenizer.pad([{"input_ids": prefix_ids}], return_tensors="pt").to(original_model.device)
            outputs_original = original_model.generate(**inputs_original, max_new_tokens=SUFFIX_LEN, do_sample=False, pad_token_id=original_tokenizer.pad_token_id)
            gen_ids_original = outputs_original[0][len(prefix_ids):].tolist()
            gen_text_original = original_tokenizer.decode(gen_ids_original)

            print(colored("Original Model Output:", 'cyan'))
            print(get_colored_diff(gt_suffix_text, gen_text_original))
            print("-" * 20)

            # edited model
            inputs_edited = edited_tokenizer.pad([{"input_ids": prefix_ids}], return_tensors="pt").to(edited_model.device)
            outputs_edited = edited_model.generate(**inputs_edited, max_new_tokens=SUFFIX_LEN, do_sample=False, pad_token_id=edited_tokenizer.pad_token_id)
            gen_ids_edited = outputs_edited[0][len(prefix_ids):].tolist()
            gen_text_edited = edited_tokenizer.decode(gen_ids_edited)
            
            print(colored("Edited Model Output:", 'cyan'))
            print(get_colored_diff(gt_suffix_text, gen_text_edited))
            print("\n" + "-"*50 + "\n")