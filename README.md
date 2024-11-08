# CS-224N-Emotion-Segmentation

### Contributors
  1.) Hannah Levin <br>
  2.) Samir Agarwala <br>
  3.) Juan Pablo Triana <br>

### Project Description
This Custom CS224N NLP final project consists on understanding complex emotions in sentences segments.

Using large language models (LLMs), specifically the LLAMA-2 70B model, we generated a dataset of 3,907 sentences, each containing 1 to 3 segments associated with distinct ground truth emotions.

We tokenized these sentences with a BERT tokenizer and passed the encoded tokens through a BERT encoder. The resulting embeddings were then input into a bidirectional LSTM, which performed sentence-emotion segmentation in alignment with the ground truth labels.

### Project Results
When benchmarked against MLP and N-gram models, this approach achieved an intersection-over-union score of 0.9230 and an emotion accuracy of 0.5021, demonstrating a substantial improvement over the baseline models. However, these results also highlight the need for further exploration into how humans convey emotions and the potential of different models to capture these subtleties more effectively.


