# recreate the activation ratio plots in the paper

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

MODEL_NAME = "PleIAs/Baguettotron"
MEMORIZED_FILE = "../data/unique_memorizations.jsonl"
CLEAN_FILE = "../data/matched_clean_set.jsonl"
KFAC_STATS_DIR = "../data/kfac_stats"
OUTPUT_DIR = "../data/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64

class WindowDataset(Dataset):
    def __init__(self, filepath, key='prefix_ids'):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.data = [json.loads(line)[key] for line in f]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}

def get_activation_hook(storage):
    def hook(model, input):
        storage['activation'] = input[0].detach()
    return hook

def compute_activation_strengths(model, dataloader, target_layer_name, eigenvectors_by_band):
    activation_storage = {}
    hook_handle = model.get_submodule(target_layer_name).register_forward_pre_hook(get_activation_hook(activation_storage))
    
    total_samples = 0
    band_strengths = {band: 0.0 for band in eigenvectors_by_band}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"processing layers", leave=False):
            inputs = batch.to(model.device)
            model(**inputs)
            
            activations = activation_storage['activation'].view(-1, activation_storage['activation'].size(-1)).to(torch.float32)
            for band, e_vecs in eigenvectors_by_band.items():
                projections = torch.matmul(activations, e_vecs)
                band_strengths[band] += torch.sum(projections**2).item()

            total_samples += activations.size(0)
    for band in band_strengths:
        if total_samples > 0:
            band_strengths[band] /= total_samples

    hook_handle.remove()    
    return band_strengths


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = 'left'; tokenizer.pad_token = "[PAD]"

mem_dataset = WindowDataset(MEMORIZED_FILE)
clean_dataset = WindowDataset(CLEAN_FILE)

def pad_collate(batch):
    return tokenizer.pad(batch, return_tensors="pt")
mem_loader = DataLoader(mem_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate)
clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate)

# map K-FAC stat file names to actual model layer names
actual_mlp_names = [name for name, mod in model.named_modules() if "mlp" in name and isinstance(mod, torch.nn.Linear)]
name_map = {name.replace('.', '_'): name for name in actual_mlp_names}
stat_files = sorted(Path(KFAC_STATS_DIR).glob("*.pt"))

results = {}
for stat_file in tqdm(stat_files, desc="analyzing Layers"):
    layer_name = name_map.get(stat_file.stem)
    if not layer_name or ('up_proj' not in layer_name and 'gate_proj' not in layer_name):
        continue
        
    stats = torch.load(stat_file, map_location=model.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(stats['A']) # eigh optimized for symmetric matrices
    eigenvectors = eigenvectors[:, torch.argsort(eigenvalues, descending=True)] # sort by largest-smallest eigenvalues
    
    dim = eigenvectors.size(0)
    bands = {
        "Top 10%": (0, int(dim * 0.10)), 
        "10-25%": (int(dim * 0.10), int(dim * 0.25)),
        "25-50%": (int(dim * 0.25), int(dim * 0.50)),
        "Bottom 50%": (int(dim * 0.50), dim)
    }
    eigenvectors_by_band = {band: eigenvectors[:, s:e] for band, (s, e) in bands.items()}
    
    mem_strengths = compute_activation_strengths(model, mem_loader, layer_name, eigenvectors_by_band)
    clean_strengths = compute_activation_strengths(model, clean_loader, layer_name, eigenvectors_by_band)
    
    results[layer_name] = {band: np.sqrt(mem_strengths[band] / clean_strengths[band]) for band in bands}

results_path = Path(OUTPUT_DIR) / "activation_ratios.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nresults saved to {results_path}")

layer_indices = sorted([int(name.split('.')[2]) for name in results.keys()])

for proj_type in ['up_proj', 'gate_proj']:
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    
    plot_data = {band: [] for band in bands}
    plot_layers = []
    
    for layer_idx in layer_indices:
        layer_name = f"model.layers.{layer_idx}.mlp.{proj_type}"
        if layer_name in results:
            plot_layers.append(layer_idx)
            for band in bands:
                plot_data[band].append(results[layer_name][band])
    
    if not plot_layers: continue

    for band, ratios in plot_data.items():
        plt.plot(plot_layers, ratios, marker='o', linestyle='-', label=band)
        
    plt.title(f"Memorized/Clean Activation Ratio for '{proj_type}' Weights", fontsize=16)
    plt.xlabel("MLP Layer", fontsize=12)
    plt.ylabel("Activation Ratio", fontsize=12)
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.legend(title="Curvature Percentile")
    plt.tight_layout()
    
    plot_path = Path(OUTPUT_DIR) / f"activation_ratio_{proj_type}.png"
    plt.savefig(plot_path)
    print(f"plot saved to {plot_path}")
    plt.show()