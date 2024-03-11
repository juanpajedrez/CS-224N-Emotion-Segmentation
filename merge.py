import json

def merge_json_files(file1, file2, file3):
    # Initialize an empty dictionary to hold merged data
    merged_data = {}

    # A list of files to iterate over
    files = [(file1, 'file1_'), (file2, 'file2_'), (file3, 'file3_')]

    for file, prefix in files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Prefix each key with the file identifier
            for key in data:
                new_key = f"{prefix}{key}"
                merged_data[new_key] = data[key]

    return merged_data

# File paths
file1 = 'data/3.7.24/data_new_50000.json'
file2 = 'data/3.8.24/data_new.json'
file3 = 'data/3.7.24/new/data_new_50000.json'

# Merge JSON files
merged_data = merge_json_files(file1, file2, file3)

# Write merged data to a new JSON file
with open('merged_data.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)
