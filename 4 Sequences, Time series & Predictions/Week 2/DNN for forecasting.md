
```python
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)

```
1. We get dataset by passing required parameters.
2. Then 3 simple layers of 10, 10 & 1 neurons respectively, input_shape as size of window and activation with 'relu'.
3. Compile and fit model as previous.

- We might pick an optimal learning rate that might be faster and more efficient.
- We can use callbacks that tweaks learning using learning rate scheduler.
- This callback will be called at end of each epoch which changes the learning rate to a values based on epoch number.
- Eg- On epoch 1, it's 1 times 1e-8 * 10 ** (1/20), and on 100 epoch, it's 1e-8 * 10 **(5)
```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
```

### Plot loss per epoch
Plot loss per epoch against epoch learning rate using following code.
```python
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```
  <img src="images/Loss%20per%20epoch.jpg" width="500">

> So we can find that 7e-6 is the learning rate at which loss is stably minimum. And we can update the learning rate in optimizer to be following.
```python
optimizer = tf.keras.optimizers.SGD(lr=7e-6, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=500, verbose=0)
```

## References
1. Deep neural network notebook - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%203.ipynb
2. DNN Forecasting Exercise - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_2_Exercise_Question.ipynb
3. DNN Forecasting Solution - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_2_Exercise_Answer.ipynb