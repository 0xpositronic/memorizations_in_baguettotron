import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import Levenshtein

BASE_MODEL_NAME = "PleIAs/Baguettotron"
EDITED_MODEL_DIR_1 = "../edited_models/baguettotron-edited-60-71-k60"
EDITED_MODEL_DIR_2 = "../edited_models/baguettotron-edited-50-75-k50"
MEMORIZED_FILE = "../data/unique_memorizations.jsonl"
CLEAN_FILE = "../data/matched_clean_set.jsonl"
OUTPUT_FILE = "../data/evaluation_results.json"

BATCH_SIZE = 32 # only used for perplexity
SUFFIX_LEN = 48

class WindowDataset(Dataset):
    def __init__(self, filepath, full_window=False):
        self.full_window = full_window
        with open(filepath, 'r') as f:
            self.data = [json.loads(line) for line in f]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.full_window:
            return {"input_ids": item['prefix_ids'] + item['suffix_ids']}
        else:
            return item

class PadCollate:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        return self.tokenizer.pad(batch, return_tensors="pt")

mem_dataset = WindowDataset(MEMORIZED_FILE, full_window=False)
clean_dataset_ppl = WindowDataset(CLEAN_FILE, full_window=True)

results = {}
for model_name in [BASE_MODEL_NAME, EDITED_MODEL_DIR_1, EDITED_MODEL_DIR_2]:
    print(f"evaluating {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'; tokenizer.pad_token = "[PAD]"; tokenizer.pad_token_id = 3

    correct_completions = 0
    total_levenshtein = 0
    with torch.no_grad():
        for item in tqdm(mem_dataset, desc="testing memorization"):
            prefix = item['prefix_ids']
            gt_suffix = item['suffix_ids']
            
            inputs = tokenizer.pad([{"input_ids": prefix}], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=SUFFIX_LEN, do_sample=False)
            generated_ids = outputs[0][inputs.input_ids.shape[1]:].tolist()

            if generated_ids == gt_suffix:
                correct_completions += 1
            
            total_levenshtein += Levenshtein.distance(generated_ids, gt_suffix)

    memorization_accuracy = (correct_completions / len(mem_dataset)) * 100
    avg_levenshtein = total_levenshtein / len(mem_dataset)
    print(f"memorization Accuracy: {memorization_accuracy:.2f}%")
    print(f"avg levenshtein distance: {avg_levenshtein:.2f}")

    pad_collate_ppl = PadCollate(tokenizer)
    clean_loader = DataLoader(clean_dataset_ppl, batch_size=BATCH_SIZE, collate_fn=pad_collate_ppl)
    
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(clean_loader, desc="calculating perplexity"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            
            non_pad_tokens = (batch["input_ids"] != tokenizer.pad_token_id).sum().item()
            tokens_in_loss = non_pad_tokens - batch["input_ids"].shape[0]

            if tokens_in_loss > 0:
                total_loss += outputs.loss.item() * tokens_in_loss
                total_tokens += tokens_in_loss

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else -1.0
    print(f"perplexity on clean set: {perplexity:.2f}")

    results[model_name] = {
        "memorization_accuracy_pct": memorization_accuracy,
        "avg_levenshtein_distance": avg_levenshtein,
        "perplexity": perplexity,
    }

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nresults saved to {OUTPUT_FILE}")