# used to visualize saved windows in the terminal

import json
from transformers import AutoTokenizer

MEMORIZED_FILE = "../data/unique_memorizations.jsonl"
MODEL_NAME = "PleIAs/Baguettotron"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

with open(MEMORIZED_FILE, 'r') as f:
    display_range = range(0, 50)
    for i, line in enumerate(f):
        if i not in display_range:
            continue
        
        data = json.loads(line)
        prefix_ids = data.get("prefix_ids")
        suffix_ids = data.get("suffix_ids")

        prefix_text = tokenizer.decode(prefix_ids)
        suffix_text = tokenizer.decode(suffix_ids)

        green_color = "\033[92m"
        reset_color = "\033[0m"

        print(f"--- Example #{i+1} ---")
        print(f"{prefix_text}{green_color}{suffix_text}{reset_color}")
        print("-----------")

print(f"\n--- examples from: {MEMORIZED_FILE} ---")