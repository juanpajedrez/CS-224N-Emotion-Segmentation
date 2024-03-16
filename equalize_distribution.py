import json
from data import EMOTION_IDX, IDX_2_EMOTION
from collections import defaultdict
import random
random.seed(10)
import numpy as np


#dict_num_segments =

# how do you plan on balancing it
# (num_segments: int in (1 2 3), emotion: str): []

with open('./data/new_final/post_processed_NO_DISGUSTED.json', 'r') as file:
    data = json.load(file)

quantiles = {
    1: 0.3,
    2: 0.025,
    3: 0.4
}


combined_data = []

for num_segs in [1, 2, 3]:


    dict_emotions = defaultdict(list)
    print("equalizing for num_segments = %d" % (num_segs))

    if num_segs == 1:
        co_mat = np.zeros(len(EMOTION_IDX))
    elif num_segs == 2:
        co_mat = np.zeros((len(EMOTION_IDX), len(EMOTION_IDX)))
    elif num_segs == 3:
        co_mat = np.zeros((len(EMOTION_IDX), len(EMOTION_IDX), len(EMOTION_IDX)))

    count_segs = 0
    final_dataset = {}
    for idx, entry in data.items():
        if len(entry["segments"]) != num_segs:
            continue
        else:
            count_segs += 1
        entry_emts = [EMOTION_IDX[segment["Emotion"].upper()] for segment in entry['segments']]
        entry_emts.sort()
        if num_segs == 1:
            co_mat[entry_emts[0]] += 1
        elif num_segs == 2:
            co_mat[entry_emts[0], entry_emts[1]] += 1
        else:
            co_mat[entry_emts[0], entry_emts[1], entry_emts[2]] += 1
    
    if num_segs == 1:
        count_emt = co_mat
    elif num_segs == 2:
        count_emt = np.sum(co_mat, axis=1) + np.sum(co_mat, axis=0)
    else:
        count_emt = np.sum(co_mat, axis=(1, 2)) + np.sum(co_mat, axis=(0, 2)) + np.sum(co_mat, axis=(0, 1))
    total_emt_count = np.sum(count_emt)

    # if count_segs * num_segs != total_emt_count:
    #     breakpoint()
    # else:
    #     print("correct number of emotions for segments of length %d" % (num_segs))

    desired_count = np.quantile(count_emt, q=quantiles[num_segs])

    keep_prob = desired_count / count_emt
    keep_prob[keep_prob > 1] = 1

    for idx, entry in data.items():
        if len(entry["segments"]) != num_segs:
            continue

        entry_emts = [segment["Emotion"].upper() for segment in entry['segments']]
        entry_emts_idx = [EMOTION_IDX[e] for e in entry_emts]

        sample_prob = 1.0
        for i in entry_emts_idx:
            sample_prob *= keep_prob[i]
        
        if random.choices([True, False], weights=[sample_prob, 1 - sample_prob], k=1)[0]:
            final_dataset[str(len(final_dataset))] = data[idx]
            combined_data.append(data[idx])

    with open(f'./data/new_final/final_data_{num_segs}.json', 'w') as f:
        json.dump(final_dataset, f, indent=6)


random.shuffle(combined_data)
combined_data_dict = {}
for i in range(len(combined_data)):
    combined_data_dict[str(i)] = combined_data[i]

with open('./data/new_final/final_data.json', 'w') as f:
    json.dump(combined_data_dict, f, indent=6)