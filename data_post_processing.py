import json
import re
import contractions

with open('data/3.2.24/before_post_processing/data_new.json', 'r') as file:
    data = json.load(file)

unique_entries = {}

# Iterate over entries and keep only unique ones
for key, entry in data.items():
    sentence = entry['full sentence']
    if sentence not in [e for e in unique_entries.values()]:
        text = re.sub(r'[^\w\s]', '', sentence)
        # Expand contractions using the contractions library
        text = contractions.fix(text)
        unique_entries[key] = entry
        for example in entry['segments']:
            example = {'full sentence': 'My brother took my favorite toy and broke it.', 'num_segments': 2, 'segments': [{'Emotion': 'Sad', 'Segment': 'My brother took my favorite toy'}, {'Emotion': 'Angry', 'Segment': '''and I'd broke it.'''}]}
            segments = example['segments']
            for segment in segments:
                print(segment)
                segment['Segment'] = re.sub(r'[^\w\s]', '', segment['Segment'])
                segment['Segment'] = contractions.fix(segment['Segment'])


# Write the unique entries back to a new JSON file with updated indices
unique_entries_with_indices = {str(index): entry for index, entry in enumerate(unique_entries.values())}

with open('data/3.2.24/no_duplicates_data.json', 'w') as file:
    json.dump(unique_entries_with_indices, file, indent=4)
