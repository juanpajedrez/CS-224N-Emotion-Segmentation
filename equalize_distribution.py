import json
from data import EMOTION_IDX
from collections import defaultdict
import random

#dict_num_segments =

# how do you plan on balancing it
# (num_segments: int in (1 2 3), emotion: str): []

with open('post_processed_FINAL.json', 'r') as file:
    data = json.load(file)

dict_emotions = defaultdict(list)

for idx, entry in data.items():
    if len(entry["segments"]) == 0 or len(entry["segments"]) > 3:
        continue
    entry_emts = [segment["Emotion"] for segment in entry['segments']]
    tmp_emt = random.sample(entry_emts, k=1)[0]
    dict_emotions[(len(entry["segments"]), tmp_emt)].append(idx)

for idx in dict_emotions:
    print(idx, len(dict_emotions[idx]))

# calculate maximum possible size of dataset
max_size_possible = min([len(x) for _, x in dict_emotions.items()])

final_dataset = {}
num_samples = 0
for key, entry in dict_emotions.items():
    sampled_indices = random.sample(dict_emotions[key], k=max_size_possible)
    for idx in sampled_indices:
        final_dataset[str(num_samples)] = data[idx]
        num_samples += 1

with open('equal_distr_data.json', 'w') as f:
    json.dump(final_dataset, f, indent=6)
    