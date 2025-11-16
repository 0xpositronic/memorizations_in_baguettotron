# find non-memorized samples with same metadata with the memorizations

import json
import random
from collections import Counter
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import Levenshtein

DATASET_NAME = "Pleias/SYNTH"
MODEL_NAME = "PleIAs/Baguettotron"
MEMORIZED_SET_FILE = "../data/unique_memorizations.jsonl"
OUTPUT_CLEAN_FILE = "../data/matched_clean_set.jsonl"

GEN_BATCH_SIZE = 2048
PREFIX_LEN, SUFFIX_LEN = 64, 48
WINDOW_LEN = PREFIX_LEN + SUFFIX_LEN
LEVENSHTEIN_THRESHOLD = 5
MIN_COMPLEXITY_RATIO = 0.4
SHUFFLE_BUFFER_SIZE = 100_000
SHUFFLE_SEED = 5

wanted_properties = Counter()
with open(MEMORIZED_SET_FILE, 'r') as f:
    for line in f:
        meta = json.loads(line)['metadata']
        wanted_properties.update([(meta['language'], meta['exercise'])])

total_needed = sum(wanted_properties.values())
print(f"finding {total_needed} matched clean examples"); print("required distribution:", wanted_properties)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"; tokenizer.pad_token = "[PAD]"
special_ids_set = set(tokenizer.all_special_ids)

used_source_ids = set() # exclude same sources
with open(MEMORIZED_SET_FILE, 'r') as f:
    for line in f: used_source_ids.add(json.loads(line)['source_id'])

dataset = load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER_SIZE)
data_iterator = iter(dataset)

batch_prefixes, batch_gt_suffixes, batch_metadata, batch_source_ids = [], [], [], []
final_clean_set = []


pbar = tqdm(total=total_needed, desc="finding matched clean examples")
with torch.no_grad():
    dataset_exhausted = False
    while not dataset_exhausted and sum(wanted_properties.values()) > 0:
        while len(batch_prefixes) < GEN_BATCH_SIZE: # build the batch
            try:
                sample = next(data_iterator)
            except StopIteration:
                dataset_exhausted = True; break
            
            source_id = sample.get('synth_id')
            sample_meta = (sample.get('language'), sample.get('exercise'))
            if source_id in used_source_ids or wanted_properties[sample_meta] <= 0:
                continue

            full_text = (f"<|im_start|>assistant\n<think>\n{sample.get('synthetic_reasoning', '')}\n"
                            f"</think>{sample.get('synthetic_answer', '')}<|im_end|>")
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(token_ids) < WINDOW_LEN: continue
            
            valid_windows = []
            for i in range(len(token_ids) - WINDOW_LEN + 1):
                suffix_ids = token_ids[i + PREFIX_LEN : i + WINDOW_LEN]
                if len(set(suffix_ids)) / SUFFIX_LEN < MIN_COMPLEXITY_RATIO or any(tok in special_ids_set for tok in suffix_ids):
                    continue
                prefix_ids = token_ids[i : i + PREFIX_LEN]
                valid_windows.append((prefix_ids, suffix_ids))

            if valid_windows:
                prefix_ids, suffix_ids = random.choice(valid_windows)
                batch_prefixes.append(prefix_ids)
                batch_gt_suffixes.append(suffix_ids)
                batch_metadata.append(sample_meta)
                batch_source_ids.append(source_id)
        if not batch_prefixes: continue
        
        inputs = tokenizer.pad([{"input_ids": p} for p in batch_prefixes], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=SUFFIX_LEN, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        start_index = inputs.input_ids.shape[1]
        for i in range(len(batch_prefixes)):
            source_id = batch_source_ids[i]
            if source_id in used_source_ids: continue # one window per document
            
            metadata_tuple = batch_metadata[i]
            if wanted_properties[metadata_tuple] <= 0: continue

            generated_ids = outputs[i][start_index:].tolist()
            gt_ids = batch_gt_suffixes[i]
            distance = Levenshtein.distance(generated_ids, gt_ids)

            if distance >= LEVENSHTEIN_THRESHOLD: # minimum diffrence (don't take generations that are only wrong by < 5 tokens)
                window_data = { "prefix_ids": gt_ids, "suffix_ids": gt_ids,
                                "metadata": {"language": metadata_tuple[0], "exercise": metadata_tuple[1]},
                                "source_id": source_id }
                final_clean_set.append(window_data)
                wanted_properties.subtract([metadata_tuple])
                used_source_ids.add(source_id)
                pbar.update(1)
        
        batch_prefixes.clear(); batch_gt_suffixes.clear(); batch_metadata.clear(); batch_source_ids.clear()

pbar.close()
with open(OUTPUT_CLEAN_FILE, 'w') as f:
    for record in final_clean_set:
        f.write(json.dumps(record) + '\n')
print(f"\nsaved {len(final_clean_set)} clean examples to '{OUTPUT_CLEAN_FILE}'")