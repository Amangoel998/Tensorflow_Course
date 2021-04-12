# Writing Poetry

```python
data = open('file.txt').read()

dimensionality_of_embedding = 100
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, dimensionality_of_embedding, input_length=max_seq_len-1),
    # We can remove Bidirectional, if we do not require poetry to make sense going backward
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
model.fit(xs, ys, epochs=100, verbose=1)
```


## References
1. https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt
2. https://www.tensorflow.org/tutorials/text/text_generation
3. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%202%20-%20Notebook.ipynb
4. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Answer.ipynb#scrollTo=BOwsuGQQY9OL