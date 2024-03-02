# !pip install together
# !python -m pip install requests

from ast import Continue
import re
import json
import requests
import time

from data import EMOTION_IDX


def parse_text(text):
    print("Parsing...")
    num_bad = 0
    lines = text.strip().split("\n")

    # Concatenate the lines to get each sentence in one row
    sentences = []
    cur_sentence = None
    sentence_emotion_dict = {}
    raw_text = []
    # print("Lines: ", lines)
    for line in lines:
        if len(line) == 0:
            continue  # ignore if nothing generated

        if "Sentence" in line:
            sentence_text = line.split("Sentence")[1].split(":")[1].strip()
            cur_sentence = sentence_text
            sentence_emotion_dict[cur_sentence] = []
            raw_text.append(line)
            print(line)

        if "Emotion" in line:
            pattern = r'<(.*?)-Start>(.*?)<(.*?)-End>'
            pairings = re.findall(pattern, line)
            # print(pairings)
            print(line)

            for pair in pairings:
                emotion = pair[1]
                emotion_label = pair[2]
                if emotion_label.upper() not in EMOTION_IDX:
                    print(f"emotion BAD: {emotion_label}")
                    raw_text.pop(-1)
                    print("Bad removed, go to next line")
                    num_bad += 1
                    continue
                sentence_emotion_dict[cur_sentence].append((emotion_label, emotion))

            annotated_concat_sentence = ''.join(item[1] for item in sentence_emotion_dict[cur_sentence])
            if cur_sentence != annotated_concat_sentence:
                raw_text.pop(-1)
                num_bad += 1
                print("Bad removed, go to next line")
                continue
            raw_text.append(line)

                # if segments dont add up to the exact sentence then continue and trash the batch
    if num_bad >= 5:
        return None, None
    print("------")
    return sentence_emotion_dict, raw_text

def dict_to_json(sentence_emotion_dict, data_file, to_save=False):

  for i, (key, value) in enumerate(sentence_emotion_dict.items()):
      num_segments = len(value)
      segments = [{"Segment": segment[1], "Emotion": segment[0]} for segment in value]
      entry_data = {
            "full sentence": key,
            "num_segments": num_segments,
            "segments": segments
        }
      data_file[len(data_file.keys())] = entry_data

      if to_save:
        with open("data/data_final.json", "w") as json_file:
            json.dump(data_file, json_file, indent=4)
              # json_file.write('\n')  # Add a newline after each entry




new_prompt = '''
[INST] We want to generate a dataset of examples for emotion segmentation in text. The only possible tags for emotions are <Angry>, <Surprised>, <Disgusted>, <Happy>, <Fearful>, <Sad>, and if there is no emotions in a segment use <Neutral>. For each emotion additionally tag it with a suffix -Start and -End. Each -Start must have a corresponding -End or the example is invalid. Here are in-context examples to learn from:

Some examples:

Sentence: Today was a bad day but at least I saw a llama.
Emotions: <Sad-Start>Today was a bad day<Sad-End><Happy-Start>but at least I saw a llama.<Happy-End>


Sentence: My teacher called on my friend in class to present and my friend *freaking* volunteered me instead.
Emotions: <Neutral-Start>My teacher called on my friend in class to presents<Neutral-End><Angry-Start>and my friend *freaking* volunteered me instead.<Angry-End>

Sentence: The pizza was delicious, but it gave me a stomachache.
Emotions: <Happy-Start>The pizza was delicious<Happy-End><Sad-Start>, but it gave me a stomachache.<Sad-End>

Sentence: I'm so done with this project... it's taking forever, but I'm close to being done and I have to admit that I'm kind of excited to see the final result.
Emotions: <Angry-Start>I'm so done with this project... it's taking forever<Angry-End><Neutral-Start>, but I'm close to being done<Neutral-End><Happy-Start> and I have to admit that I'm kind of excited to see the final result.<Happy-End>

Sentence: I can't believe I found a cockroach in the shower... it totally grossed me out.
Emotions: <Surprise-Start> I can't believe I found a cockroach in the shower<Surprise-End><Disgust-Start... it totally grossed me out.<Disgust-End>

ONLY output exactly ten additional unique examples using this format and using ONLY 1-3 emotion tags of <Angry>, <Surprised>, <Disgusted>, <Happy>, <Fearful>, <Sad> per sentence.   [/INST]"
'''
endpoint = 'https://api.together.xyz/v1/chat/completions'

data_file = {}
num_iters = 0

raw_data = []
total_generated_examples = 1700  # we expect 100 total examples
sentences_per_batch = 10

range_total = total_generated_examples // sentences_per_batch
for batch_num in range(10):
    print("new prompt: ", new_prompt)
    print("-----------")
    print(f"Batch {batch_num} / {range_total}")
    res = requests.post(endpoint, json={
        "model": "meta-llama/Llama-2-70b-chat-hf",
        "max_tokens": 2000,
        "prompt": new_prompt,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "[/INST]",
            "</s>"
        ],
        "repetitive_penalty": 1,
        "update_at": "2024-02-28T08:59:13.529Z"
    }, headers={
        "Authorization": "Bearer f02ad1986b51f7b84aac3379ccbf23f9ae555caf720003d7c322ad32cda12531",
    })

    num_iters += 1
    print("request to llama made")

    data = res.json()
    # Extract text from the choices
    text = data["choices"][0]["message"]["content"]

    with open("data/raw_data_final.txt", "a") as fout:
        fout.write(text)
        fout.write('\n')
        fout.close()

    try:
        sentence_emotion_dict, raw_text = parse_text(text)
    except:
        continue

    if sentence_emotion_dict == None:
        continue

    if len(raw_text) != 0:
        raw_string = '\n'.join(raw_text)
        # convert list of strings to str + \n + str

        new_prompt = f'''
[INST] We want to generate a dataset of examples for emotion segmentation in text. The only possible tags for emotions are <Angry>, <Surprised>, <Disgusted>, <Happy>, <Fearful>, <Sad>, and if there is no emotions in a segment use <Neutral>. For each emotion additionally tag it with a suffix -Start and -End. Each -Start must have a corresponding -End or the example is invalid. Here are in-context examples to learn from:

Some examples:

{raw_string}

ONLY output exactly ten additional unique examples using this format and using ONLY 1-3 emotion tags of <Angry>, <Surprised>, <Disgusted>, <Happy>, <Fearful>, <Sad> per sentence.   [/INST]"
'''

    raw_data.append(raw_text)

    dict_to_json(sentence_emotion_dict, data_file, to_save=(num_iters % 10 == 0))
    # print(sentence_emotion_dict)
    # print("----------")
    time.sleep(2)

with open("data/data_final.json", "a") as json_file:
    json.dump(data_file, json_file, indent=4)

print("done")