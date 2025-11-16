# find memorizations in thinking+answer part of each document
# takes a long time find memorizations so this runs until interrupted
# can continue using checkpoints. I ran this for ~19 hours on a 3090 to get ~600 unique samples
import json
import os
import time
from pathlib import Path
import fcntl
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_NAME = "Pleias/SYNTH"
MODEL_NAME = "PleIAs/Baguettotron"

OUTPUT_MEMORIZED_FILE = "../data/raw_memorizations.jsonl"
CHECKPOINT_FILE = "../data/mem_checkpoint.json"

GEN_BATCH_SIZE = 2048
PREFIX_LEN, SUFFIX_LEN = 64, 48
WINDOW_LEN = PREFIX_LEN + SUFFIX_LEN

MIN_COMPLEXITY_RATIO = 0.4

def append_to_jsonl(filepath, data):
    filepath = Path(filepath); filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try: f.write(json.dumps(data) + '\n'); f.flush(); os.fsync(f.fileno())
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE): return {'samples_processed': 0, 'windows_tested': 0, 'windows_filtered': 0}
    with open(CHECKPOINT_FILE, 'r') as f: return json.load(f)

def save_checkpoint(state):
    state['last_update'] = time.time()
    tmp_file = CHECKPOINT_FILE + '.tmp'
    with open(tmp_file, 'w') as f: json.dump(state, f, indent=2)
    os.replace(tmp_file, CHECKPOINT_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"; tokenizer.pad_token = "[PAD]"

special_ids_set = set(tokenizer.all_special_ids)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
checkpoint = load_checkpoint(); samples_to_skip = checkpoint['samples_processed']

dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
if samples_to_skip > 0: dataset = dataset.skip(samples_to_skip)
data_iterator = iter(dataset)

pbar = tqdm(initial=samples_to_skip)
try:
    with torch.no_grad():
        dataset_exhausted = False
        while not dataset_exhausted:
            batch_start_time = time.time()
            batch_candidates = []
            
            while len(batch_candidates) < GEN_BATCH_SIZE:
                try:
                    sample = next(data_iterator)
                    pbar.update(1)
                except StopIteration:
                    dataset_exhausted = True; break
                
                token_ids = tokenizer.encode((f"<|im_start|>assistant\n<think>\n{sample.get('synthetic_reasoning', '')}\n"
                                              f"</think>{sample.get('synthetic_answer', '')}<|im_end|>"), add_special_tokens=False)
                
                if len(token_ids) < WINDOW_LEN: continue
                
                i = 0
                while i <= len(token_ids) - WINDOW_LEN:
                    suffix_ids = token_ids[i + PREFIX_LEN : i + WINDOW_LEN]
                    
                    if len(set(suffix_ids)) / SUFFIX_LEN < MIN_COMPLEXITY_RATIO or \
                       any(tok in special_ids_set for tok in suffix_ids):
                        checkpoint['windows_filtered'] += 1; i += 1; continue

                    prefix_ids = token_ids[i : i + PREFIX_LEN]
                    batch_candidates.append({
                        "prefix_ids": prefix_ids, "suffix_ids": suffix_ids,
                        "metadata": {"language": sample.get('language'), "exercise": sample.get('exercise')},
                        "source_id": sample.get('synth_id')
                    })
                    
                    i += SUFFIX_LEN
                    # this was a really bad choice as we had to deduplicate long repeating sequences
                    # we should have sampled one window per document like in the paper
                    if len(batch_candidates) >= GEN_BATCH_SIZE: break
            
            if not batch_candidates: continue
            
            inputs = tokenizer.pad([{"input_ids": c['prefix_ids']} for c in batch_candidates], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=SUFFIX_LEN, do_sample=False, pad_token_id=tokenizer.pad_token_id)

            start_index = inputs.input_ids.shape[1]
            for i, candidate in enumerate(batch_candidates):
                generated_ids = outputs[i][start_index:].tolist()
                if generated_ids == candidate['suffix_ids']:
                    append_to_jsonl(OUTPUT_MEMORIZED_FILE, candidate)
            
            batch_duration = time.time() - batch_start_time
            wps = len(batch_candidates) / batch_duration if batch_duration > 0 else float('inf')
            checkpoint['samples_processed'] = pbar.n
            checkpoint['windows_tested'] += len(batch_candidates)
            if pbar.n % 10000 == 0: save_checkpoint(checkpoint)
                
            mem_count = sum(1 for _ in open(OUTPUT_MEMORIZED_FILE,'r')) if os.path.exists(OUTPUT_MEMORIZED_FILE) else 0
            pbar.set_description(f"Scanned: {pbar.n:,} | Tested: {checkpoint['windows_tested']:,} | Mem Found: {mem_count} | Speed: {wps:.1f} w/s")

except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    save_checkpoint(checkpoint)
    pbar.close()
    mem_count = sum(1 for _ in open(OUTPUT_MEMORIZED_FILE, 'r')) if os.path.exists(OUTPUT_MEMORIZED_FILE) else 0
    print(f"Total raw memorized candidates found: {mem_count}")
    print(f"Total source samples processed: {checkpoint.get('samples_processed', 0):,}")