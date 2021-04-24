# Predicting Time series using RNN

- A recurrent neural network contains multiple Dense and recurrent layers which has flexibility to process any sequence
  of input.
- We can feed batches of sequences and get batches of forecasts.
- In earlier examples, the input shape has 2 dimensions
    1. Batch dimension
    2. All the input features
- The full input shape used when using RNNs is 3-dimensional:
    1. Batch size
    2. No. of time steps
    3. No. of dimensionality of input at each step.
        - For univariate, this values will be 1
        - For multivariate, this value will be more.

# Details for Recurrent Layers

- There's only 1 cell which is used repeatedly to compute the outputs.
- At each time step, memory cell takes input value for next cell, here 0.
- It calculates output for that step, Y0, and the state vector, H0, is fed into next step.
- Hence, the name recurrent, because the values recur due to output of cell and one step is fed back into the cell at
  next step.
- So a location of word in sentence can determine semantics.
  <img src="images/Recurrent%20Layers.jpg" width="500">

# Shape of the inputs to the RNN

- For example, we have a window of 30 time steps, we are batching them in sizes of 4, the shape will be 4 * 30 * 1.
- At each time step, the memory cell input will be four by one matrix.
- The cell will also take input of state matrix from previous step, initially 0, then H0, H1....

- If memory cell is comprised of 3 neurons, output matrix will be 4 * 3, as batch size is 4 and no. of neurons is 3.
- Hence, the full output of layer is 3-dimensional, ie, 4 * 30 * 3.

  <img src="images/Recurrent%20Layers%20input%20shape.jpg" width="500">
- In simple RNN, state output H, is copy of output matrix Y, ie. H0 is copy of Y0 and so on.
- If we do not want a sequence output, we want to get single vector for each instance of batch.
- This is called **Sequence to vector RNN**. We simply ignore all outputs except the last one..

> In keras, if you want to return sequence as output not just last vector output, set return_sequences=True.

  <img src="images/Sequence%20to%20vector%20RNN.jpg" width="500">

## RNN for sequence of inputs example

```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])
])
```

  <img src="images/Example%20RNN.jpg" width="500">

## Lambda Type Layer

- These type of layers allows to perform arbitrary operations to effectively expand the functionality of Tensorflow's
  keras.
- The window dataset helper function, returns 2-dimensional batches of widows of data.
    1. Batch size.
    2. No. of time steps.
- But to get 3-dimensional input shape for RNN, we can use Lambda function and expand the dimensions.
- By multiplying output by 100, we can get values in 10th place can help us in learning.
- It's because the tanh activation function only returns values b/w 1 & 0.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0),
])
])
```

## Prediction using RNN and optimizations

- Created a callback to change the learning from 1e-8 to 1e-6.
- Huber loss function that's less sensitive to outliers as the data can get a little noisy.

- The best learning rate came out at around 5e-5
- After training for 400 epochs, the result became unstable so it is better to train only for 400 epochs.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=400)
```

  <img src="images/Forecast%20data.jpg" width="500">

## Prediction using LSTM

- The state in RNNs is factor of subsequent calculations but its impact can diminish over timestamps.
- LSTM are the cell state to this that keep state throughout the life of training, so state is passed from cell to cell
  and timestamps too timestamps.
- The state can also be bidirectional, so state moves forward and backwards.
- Bidirectional is helpful in sentence sequence but may not be so in numeric sequence.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```

  <img src="images/2%20layer%20bidirectional%20LSTM.jpg" width="500">

#  

## References

1. Huber Loss - https://en.wikipedia.org/wiki/Huber_loss
2. Forecasting using only RNN
   - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%202%20-%20RNN.ipynb
3. LSTM lesson from Andrew ng - https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay
4. LSTM Notebook
   - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%204%20-%20LSTM.ipynb
5. MAE Exercise Answer notebook
   - https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Exercise%20Answer.ipynb