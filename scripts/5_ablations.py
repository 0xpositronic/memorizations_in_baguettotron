import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_NAME = "PleIAs/Baguettotron"
KFAC_STATS_DIR = "../data/kfac_stats"
EDITED_MODEL_DIR = "../edited_models/baguettotron-edited-60-71-k60"
Path(EDITED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
# I didn't do a hyperparameter seach on these but this section is where we see a visible divergence between different eigenvector bands in the plots
LAYERS_TO_EDIT = range(60, 71)
PROJECTIONS_TO_EDIT = ['gate_proj', 'up_proj']
PERCENT_CURVATURE_TO_KEEP = 0.60 # same as paper

# idk if loading in float32 makes much diffence here
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32, device_map="cpu").eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

actual_mlp_names = [name for name, mod in model.named_modules() if "mlp" in name and isinstance(mod, torch.nn.Linear)]
name_map = {name.replace('.', '_'): name for name in actual_mlp_names}

kfac_data = {}
stat_files = sorted(Path(KFAC_STATS_DIR).glob("*.pt"))
for stat_file in tqdm(stat_files, desc="loading stats"):
    layer_name = name_map.get(stat_file.stem)

    stats = torch.load(stat_file, map_location='cpu')

    eivals_a, eivecs_a = torch.linalg.eigh(stats['A'])
    eivals_g, eivecs_g = torch.linalg.eigh(stats['G'])
    
    # sort eigenvalues and eigenvectors in descending order
    kfac_data[layer_name] = {'eivals_a': eivals_a.flip(dims=(0,)), 'eivecs_a': eivecs_a.flip(dims=(1,)),
                             'eivals_g': eivals_g.flip(dims=(0,)), 'eivecs_g': eivecs_g.flip(dims=(1,))}
    
for layer_idx in tqdm(LAYERS_TO_EDIT, desc="editing layers"):
    for proj_name in PROJECTIONS_TO_EDIT:
        layer_name = f"model.layers.{layer_idx}.mlp.{proj_name}"

        W = model.get_submodule(layer_name).weight.data
        data = kfac_data[layer_name]
        U_A, U_G = data['eivecs_a'], data['eivecs_g']
        eivals_a, eivals_g = data['eivals_a'], data['eivals_g']

        C = U_G.T @ W @ U_A #  project W into the K-FAC basis

        curvature_mass_matrix = torch.outer(eivals_g, eivals_a)
        all_masses = curvature_mass_matrix.flatten()
        
        total_mass = torch.sum(all_masses)

        sorted_masses, _ = torch.sort(all_masses, descending=True)
        cumulative_mass = torch.cumsum(sorted_masses, dim=0)
        
        # find the index where cumulative mass just exceeds our target percentage
        threshold_idx = (cumulative_mass >= total_mass * PERCENT_CURVATURE_TO_KEEP).nonzero(as_tuple=True)[0][0]
        mass_threshold = sorted_masses[threshold_idx]
        
        mask = (curvature_mass_matrix >= mass_threshold).float()

        C_masked = C * mask
        W_edited = U_G @ C_masked @ U_A.T # project back using masked C
        
        model.get_submodule(layer_name).weight.data = W_edited

model.save_pretrained(EDITED_MODEL_DIR)
tokenizer.save_pretrained(EDITED_MODEL_DIR)

edit_info = {
    "base_model": BASE_MODEL_NAME,
    "layers_edited": list(LAYERS_TO_EDIT),
    "projections_edited": PROJECTIONS_TO_EDIT,
    "percent_curvature_kept": PERCENT_CURVATURE_TO_KEEP,
    "kfac_stats_source": KFAC_STATS_DIR,
}

with open(Path(EDITED_MODEL_DIR) / "edit_info.json", 'w') as f:
    json.dump(edit_info, f, indent=2)
print(f"edited model saved to {EDITED_MODEL_DIR}")