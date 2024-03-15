import json

with open('./data/final/final_data.json', 'r') as f:
    data = json.load(f)

num_words = 0
num_sentences = 0
for sample in data:
    sentence = data[sample]["full sentence"]
    word_count_sentence = len(sentence.strip().split())
    num_words += word_count_sentence
    num_sentences += 1

print("Avg number of words per sentence: %d" % (num_words / num_sentences))