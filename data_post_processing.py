import json
import re
import contractions

with open('data/3.2.24/before_post_processing/data_new.json', 'r') as file:
    data = json.load(file)

unique_entries = {}

# Iterate over entries and keep only unique ones
for key, entry in data.items():
    sentence = entry['full sentence']
    if entry["num_segments"] == 0:
        continue
    if sentence not in [e for e in unique_entries.values()]:
        entry['full sentence'] = re.sub(r'[^\w\s]', '', sentence)
        # Expand contractions using the contractions library
        entry['full sentence'] = contractions.fix(entry['full sentence'], slang=False)
        unique_entries[key] = entry
        for example in entry['segments']:
            example['Segment'] = re.sub(r'[^\w\s]', '', example['Segment'])
            example['Segment'] = contractions.fix(example['Segment'], slang=False)


# Write the unique entries back to a new JSON file with updated indices
unique_entries_with_indices = {str(index): entry for index, entry in enumerate(unique_entries.values())}

with open('data/3.2.24/no_duplicates_data.json', 'w') as file:
    json.dump(unique_entries_with_indices, file, indent=4)
