import json
from data import EMOTION_IDX
import matplotlib.pyplot as plt
from collections import defaultdict

# Open the JSON file
with open('test_data_1.json') as f:
    data = json.load(f)

# Initialize a list to store all values of the "emotion" key

emotion_labels = list(EMOTION_IDX.keys())


emotion_dict = {key: 0 for key in emotion_labels}



# data on emotions across segments and frequency of these emotions per segment

unique_emotions_per_segment = defaultdict(int)
num_segments_per_sentence = defaultdict(int)
# Iterate through each entry in the JSON data

for key, entry in data.items():
    # Check if the "emotion" key exists in the entry
    segments = entry['segments']
    num_segments = len(segments)
    num_segments_per_sentence[str(num_segments)] += 1
    cur_emotions = []
    for segment in segments:
        cur_emotion = segment["Emotion"].upper()
        cur_emotions.append(cur_emotion)
        # Add the value of the "emotion" key to the list
        emotion_dict[cur_emotion] += 1
    unique_emotions_per_segment[str(len(set(cur_emotions)))] += 1

print(emotion_dict) # must have 1428 of each emotion token
print(unique_emotions_per_segment)

# Print all values of the "emotion" key
plt.rcParams.update({'font.size': 14})  # You can adjust the size as needed
plt.bar(emotion_dict.keys(), emotion_dict.values())
plt.xlabel('Emotions Present in Segments', fontsize=14)  # Increase font size
plt.ylabel('Frequency', fontsize=14)  # Increase font size
plt.title('Histogram', fontsize=16)  # Increase font size
plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels and increase font size
plt.yticks(fontsize=12)  # Increase font size of y-axis ticks



denom = sum([x for x in emotion_dict.values()])
emotion_counts = {key: emotion_dict[key] / denom * 100.0 for key in emotion_dict.keys() }



for emotion, value in emotion_dict.items():
    plt.text(emotion, value, f'{emotion_counts[emotion]:.2f}%', ha='center', va='bottom', fontsize=12)


total_values = sum(unique_emotions_per_segment.values())

# Calculate percentage of each value
percentage_values = {key: (value / total_values) * 100 for key, value in unique_emotions_per_segment.items()}

print("Percentage of different values:")
for key, value in percentage_values.items():
    print(f"{key}: {value:.2f}%")


plt.tight_layout()
plt.show()




# Plotting
plt.bar(unique_emotions_per_segment.keys(), unique_emotions_per_segment.values())
plt.xlabel('Total Unique Emotions Per Example', fontsize=16)  # Adjusting font size
plt.ylabel('Frequency', fontsize=16)  # Adjusting font size
plt.title('Histogram', fontsize=18)  # Adjusting font size




denom_seg = sum([x for x in num_segments_per_sentence.values()])
emotion_counts = {key: num_segments_per_sentence[key] / denom_seg * 100.0 for key in num_segments_per_sentence.keys()}



for emotion, value in emotion_counts.items():
    plt.text(emotion, value, f'{emotion_counts[emotion]:.2f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()



plt.bar(num_segments_per_sentence.keys(), num_segments_per_sentence.values())
plt.xlabel('Total Segments per Sentence (not necessarily unique emotions)')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.tight_layout()
plt.show()


