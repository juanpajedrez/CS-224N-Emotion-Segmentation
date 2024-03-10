import json
import re
import contractions

with open('merged_data.json', 'r') as file:
    data = json.load(file)

unique_entries = {}

# Iterate over entries and keep only unique ones
for key, entry in data.items():
    sentence = entry['full sentence']
    if entry["num_segments"] == 0:
        continue


    if sentence not in [e for e in unique_entries.values()]:
        # Expand contractions using the contractions library
        entry['full sentence'] = contractions.fix(entry['full sentence'], slang=False)
        # remove punctuation after contraction
        entry['full sentence'] = re.sub(r'[^\w\s]', '', entry['full sentence'])
        entry['full sentence'] = entry['full sentence'].lower().strip()


        # print("true sentence: ", cur_sentence)
        # print("annotated version: ", annotated_concat_sentence)

        skip_example = False
        for example in entry['segments']:
            if example['Emotion'].lower() == 'disgusted':
                skip_example = True
                break
            example['Segment'] = contractions.fix(example['Segment'], slang=False)
            example['Segment'] = re.sub(r'[^\w\s]', '', example['Segment'])
            example['Segment'] = example['Segment'].lower().strip() # for some reason the strip, turns string into lists of strings for each stripped word between whitespace

        if skip_example:
            continue

        annotated_sentence = ' '.join(segment["Segment"] for segment in entry['segments'])

        if entry['full sentence'] == annotated_sentence:
            unique_entries[key] = entry


# Write the unique entries back to a new JSON file with updated indices
unique_entries_with_indices = {str(index): entry for index, entry in enumerate(unique_entries.values())}

# .strip()
# remove duplicates (make sure key and annotate dsentence is same)
# make everything lowercase
with open('post_processed_NO_DISGUSTED.json', 'w') as file:
    json.dump(unique_entries_with_indices, file, indent=4)
