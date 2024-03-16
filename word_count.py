import json
import numpy as np

with open('./data/new_final/final_data.json', 'r') as f:
    data = json.load(f)

num_words = 0
num_sentences = 0
min_words = float('inf')
max_words = 0
word_list = []

sentences_unique = set()

for sample in data:
    sentence = data[sample]["full sentence"]
    word_count_sentence = len(sentence.strip().split())
    num_words += word_count_sentence
    num_sentences += 1
    min_words = min(min_words, word_count_sentence)
    max_words = max(max_words, word_count_sentence)
    word_list.append(word_count_sentence)
    sentences_unique.add(sentence)

print("Avg number of words per sentence: %d" % (num_words / num_sentences))
print("Min number of words: %d" % (min_words))
print("Max number of words: %d" % (max_words))
print("Median number of words: %d" % (np.median(np.array(word_list))))
print("Number of unique sentences: %d" % (len(sentences_unique)))
print("Number of total sentences: %d" % (len(data)))