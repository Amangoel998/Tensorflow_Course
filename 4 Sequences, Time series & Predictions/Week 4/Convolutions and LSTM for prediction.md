## Prediction using LSTM and Convolutions

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        strides=1,
        padding='casual',
        activation='relu',
        input_shape=[None, 1]
    ),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 200)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=500)
```

- We expand the dimension of the helper function to expand dimension of input shape.

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    .window(window_size + 1, shift=1, drop_remainder=True)
    .flat_map(lambda window: window.batch(window_size + 1))
    .shuffle(buffer_size=shuffle_buffer)
    .map(lambda window: (window[:-1], window[-1:]))
    .batch(batch_size).prefetch(1)


return dataset
```

- We can tried to introduce 2 bidirectional lstm layers but this seems to be overfitting the data.
- We can also experiment to tweak with batch sizes, as smaller batch sizes seems to introduce more spikes in Mae vs loss
  graph.

## References

1. Convolutions and LSTM Notebook
   - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%201.ipynb#scrollTo=ok8LjNbbkig4
2. Real Time Time series data for sunspots - https://www.kaggle.com/robervalt/sunspots
3. Notebook for predicting Sunspots
   - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%205.ipynb#scrollTo=MD2kyYUVt3O0
4. Practice Exercise
   - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Question.ipynb
5. Exercise Solution
   - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb