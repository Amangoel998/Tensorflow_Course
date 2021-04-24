# Subwords Texts

- Sub-word meanings are often nonsensical and it's only when we put them together in sequences that they have meaningful
  semantics. Thus, some way from learning from sequences would be a great way forward.

```python
import tensorflow_datasets as tfds

imdb, infp = tfds.load9
'imdb_reviews/subwords8k', with_info = True.as_supervised = True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder

print(tokenizer.subwords)

sample_string = "Tensorflow, from basics to Advanced"
tokenized_string = tokenizer.encode(sample_string)
original_string = tokenizer.decode(tokenized_string)

# Individual decode
for ts in tokenized_string:
    print(ts, tokenizer.decode([ts]))

```

- We use GlobalAveragePooling1D instead of Flatten in model layers for sub-words.
- Using Flatten here will cause tensorflow to crash.

## References

- https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Text
- https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb