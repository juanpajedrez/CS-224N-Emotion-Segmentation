# CS-224N-Emotion-Segmentation
This Custom CS224N NLP final project consists on understanding complex emotions in sentences segments.

Using LLMs, specifically LLAMA-2 70B model, we generated a large scale dataset of 3907 sentences. Each of them
contained either 1, 2, or 3 segments; corresponding to a ground truth emotion.

Here, by usinh a BERT tokenizer, BERT encoder, we passed the encoded positional words to a bidirectional LSTM; where
it would perform sentence-emotion segmentation againts the ground truth. 

After comparing against MLP, and N-grams models, we found this approach to have an intersection over union of 0.9230; 
and a emotion accuracy of 0.5021. This showcases the massive improvement against the baselines, yet the need to further
understand how humans convey emotion; and which models would be useful to do this.
