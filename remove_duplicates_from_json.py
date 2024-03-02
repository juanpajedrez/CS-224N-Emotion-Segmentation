import json

with open('data/data_new.json', 'r') as file:
    data = json.load(file)

unique_entries = {}

# Iterate over entries and keep only unique ones
for key, entry in data.items():
    sentence = entry['full sentence']
    if sentence not in [e['full sentence'] for e in unique_entries.values()]:
        unique_entries[key] = entry

# Write the unique entries back to a new JSON file with updated indices
unique_entries_with_indices = {str(index): entry for index, entry in enumerate(unique_entries.values())}

with open('no_duplicates_data.json', 'w') as file:
    json.dump(unique_entries_with_indices, file, indent=4)
