# calculate K-FAC stats over 20M token data stream

import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_NAME = "Pleias/SYNTH"
MODEL_NAME = "PleIAs/Baguettotron"
OUTPUT_DIR = "../data/kfac_stats"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TARGET_TOKENS = 20_000_000 # same as paper (tried 40M nothing changes)
SHUFFLE_BUFFER_SIZE = 50_000 # idk
SHUFFLE_SEED = 42
CHUNK_SIZE = 512 # same as paper
BATCH_SIZE = 16 # 22.36 GB VRAM

class KFAC_dataset(IterableDataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __iter__(self):
        streamed_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
        shuffled_stream = streamed_dataset.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER_SIZE)
        
        token_buffer = []
        total_tokens_processed = 0
        for sample in shuffled_stream:
            if total_tokens_processed >= TARGET_TOKENS: break
            
            formatted_text = (f"<|im_start|>user\n{sample.get('query', '')}<|im_end|>\n"
                              f"<|im_start|>assistant\n<think>\n{sample.get('synthetic_reasoning', '')}\n"
                              f"</think>{sample.get('synthetic_answer', '')}<|im_end|>")

            token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            token_buffer.extend(token_ids)
            total_tokens_processed += len(token_ids)
            
            while len(token_buffer) >= CHUNK_SIZE:
                chunk = token_buffer[:CHUNK_SIZE]
                token_buffer = token_buffer[CHUNK_SIZE:]
                yield torch.tensor(chunk, dtype=torch.long)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).train()
dataloader = DataLoader(KFAC_dataset(tokenizer), batch_size=BATCH_SIZE)

mlp_layers = {}
for name, module in model.named_modules():
    if "mlp" in name and isinstance(module, torch.nn.Linear):
        mlp_layers[name] = {'module': module}
        module.register_forward_pre_hook(lambda m, inp, name=name: mlp_layers[name].update({'activations': inp[0].detach()}))
        module.register_full_backward_pre_hook(lambda m, g_out, name=name: mlp_layers[name].update({'gradients': g_out[0].detach()}))

cov_stats = {}
for name, data in mlp_layers.items():
    in_dim = data['module'].in_features
    out_dim = data['module'].out_features
    cov_stats[name] = {'A': torch.zeros((in_dim, in_dim), dtype=torch.float32, device='cpu'),
                       'G': torch.zeros((out_dim, out_dim), dtype=torch.float32, device='cpu'),
                       'n_samples': 0}

pbar = tqdm(total=TARGET_TOKENS, desc="calculating K-FAC stats")
tokens_processed = 0
for batch in dataloader:
    batch = batch.to(model.device)
    inputs = batch[:, :-1]
    outputs = model(inputs)
    logits = outputs.logits

    # targets samples from models distribution
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sampled_targets = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).squeeze(-1)
    
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), sampled_targets)
    loss.backward()
    
    for name, data in mlp_layers.items():
        a = data['activations'].reshape(-1, data['activations'].size(-1)).to(torch.float32)
        g = data['gradients'].reshape(-1, data['gradients'].size(-1)).to(torch.float32)
        
        cov_stats[name]['A'] += torch.matmul(a.T, a).cpu()
        cov_stats[name]['G'] += torch.matmul(g.T, g).cpu()
        cov_stats[name]['n_samples'] += a.size(0)

    model.zero_grad() # clear before next batch
    for name, data in mlp_layers.items():
        data['activations'] = None
        data['gradients'] = None
    
    batch_tokens = batch.numel()
    tokens_processed += batch_tokens
    pbar.update(batch_tokens)    
pbar.close()

for name, stats in cov_stats.items(): # saving
    n = stats['n_samples']        
    stats['A'] /= n
    stats['G'] /= n
    
    safe_name = name.replace('.', '_')
    filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.pt")
    torch.save({'A': stats['A'], 'G': stats['G'], 'n_samples': n}, filepath)
print(f"K-FAC stats saved to '{OUTPUT_DIR}'")
print(f"{tokens_processed:,} tokens processed")