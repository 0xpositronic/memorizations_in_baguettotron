# filter similar memorizations with max jaccard similarity threshold

import json
from tqdm import tqdm

RAW_MEMORIZED_FILE = "../data/raw_memorizations.jsonl"
FINAL_MEMORIZED_FILE = "../data/unique_memorizations.jsonl"
SIMILARITY_THRESHOLD = 0.8

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

with open(RAW_MEMORIZED_FILE, 'r') as f:
    raw_windows = [json.loads(line) for line in f]

selected_windows, selected_window_token_sets = [], []

first_window = raw_windows[0]
selected_windows.append(first_window)
selected_window_token_sets.append(set(first_window['prefix_ids'] + first_window['suffix_ids']))

for new_window in tqdm(raw_windows[1:], desc="clustering and selecting"):
    new_window_token_set = set(new_window['prefix_ids'] + new_window['suffix_ids'])
    
    is_duplicate = False
    for existing_set in selected_window_token_sets:
        similarity = jaccard_similarity(new_window_token_set, existing_set)
        if similarity > SIMILARITY_THRESHOLD:
            is_duplicate = True
            break
    
    if not is_duplicate:
        selected_windows.append(new_window)
        selected_window_token_sets.append(new_window_token_set)

print(f"{len(raw_windows)} > {len(selected_windows)} removed {len(raw_windows) - len(selected_windows)}")

with open(FINAL_MEMORIZED_FILE, 'w') as f:
    for record in selected_windows:
        f.write(json.dumps(record) + '\n')
print(f"\nuniques saved to '{FINAL_MEMORIZED_FILE}'")

