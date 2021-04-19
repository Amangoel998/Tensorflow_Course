## Similar RNN like used in images
- Now the words will be grouped in size of the filter(here 5).
- Convolutions will be learned that can map word classification to desired output.
- We have 128 filters each for 5 words.
- So if max_length was 120 words and filter of 5 words, it will discard 2 words from back and front, leaving with 116 words shape sequences.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedded_dim, input_length=max_length),
    tf.keras.layers.Conv2D(128, 5, activation='relu')
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
**Note:**
If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?
(None, 116, 128)


## Gated Recurrent Units (GRU)
- To use GRU in tensorflow, just replace LSTM with GRU in model layer from previous code.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Reference
1. https://nlp.stanford.edu/projects/glove/