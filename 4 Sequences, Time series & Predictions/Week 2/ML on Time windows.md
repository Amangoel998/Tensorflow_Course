# Machine Learning on Time Windows
- First we have to divide the data into features and labels.
- Here our feature is effective a number of values in series with label being next value.
- That number of values can be called window size.
- So for 30 days at a time, we use 30 values as feature and next value as label.
- Then over time, we will train model to match 30 features on single label.
  
  
  <img src="images/ML%20on%20Time%20window.jpg" width="500">

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
# Use window to expand our dataset using windowing
dataset = dataset.window(5, shift=1, drop_remainder=True)

for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()

# 0 1 2 3 4
# 1 2 3 4 5
# 2 3 4 5 6
# 3 4 5 6 7
# 4 5 6 7 8
# 5 6 7 8 9

# Below if drop_remainder=False (default)
# 6 7 8 9
# 7 8 9
# 8 9
# 9

dataset = dataset.flat_map(lambda window: window.batch(5))
# If we try to print dataset[1].numpy() we get [1 2 3 4 5]

# Next we split data into features & labels
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
    print(x.numpy(), y.numpy())
# [0 1 2 3] [4]
# [1 2 3 4] [5]
# [2 3 4 5] [6]
# [3 4 5 6] [7]
# [4 5 6 7] [8]
# [5 6 7 8] [9]

dataset = dataset.shuffle(buffer_size=10)
# We can batch the data into set of 2
dataset = dataset.batch(2).prefetch(1)

# x = [[4 5 6 7], [1 2 3 4]]
# y = [[8], [5]]
# x = [[0 1 2 3], [3 4 5 6]]
# y = [[4], [7]]
# x = [[5 6 7 8], [2 3 4 5]]
# y = [[9], [6]]

```
## Sequence bias
Sequence bias is when the order of things can impact the selection of things. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up. 

## Feeding windowed dataset into neural network
```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
                     .window(window_size+1, shift=1, drop_remainder=True)
                     .flat_map(lambda window: window.batch(window_size+1))
                     .shuffle(buffer_size=shuffle_buffer)
                     .map(lambda window: (window[:-1], window[-1:]))
                     .batch(batch_size).prefetch(1)
    return dataset

# We split data into training & validatio n set at step 1000.
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Code to do simple linear regression
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer)
layer0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.Sequential([layer0])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizer.SGD(lr=1e-6),
    momentum=0.9
)
model.fit(dataset, epochs=100, verbose=0)
```
> After Linear regression is done, the array is split into 2 arrays, 1st array with 20 values to fit the weight values best it can which is value for x, and 2nd array is b value which is the bias.

`Y = W0*X0 + W1*X1 + W2*X2.... + WnXn + b`

Hence, to predict the values using linear regression, we can use the following
- np.newaxis reshapes the input to the input dimension used by model
```python
print("Weights".format(layer0.get_weights()))

model.predict(series[1:21][np.newaxis])
```
### Prediction Output:
<img src="images/predciction%20values.jpg" width="500">

1. Top array is the 20 values are provided to model as input.
2. Bottom is the predicted value back form out model.

### Plotting forecast on time series
```python
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time.time + window_size][np.newaxis]))
forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]
```

<img src="images/Forecast%20prediction.jpg" width="500">

### Calculate Mean Absolute Error
```python
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
# 4.9526777
```

## References
1. Creating dataset for prediction using our ML model - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%201.ipynb
2. Single layer neural network notebook - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%202.ipynb