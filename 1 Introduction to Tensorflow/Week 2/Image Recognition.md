Machine Learning depends on having good data to train a system with. In this video you saw a scenario for training a system to recognize fashion images. The data comes from a dataset called Fashion MNIST, and you can learn more about it and explore it in GitHub here. In the next video, you’ll see how to load that data and prepare it for training. You saw how the image is represented as a 28x28 array of greyscales, and how its label is a number. 

Fasion_mnist is avaiable as API call in tensorflow

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.dataset.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),     # Can use 512 layers
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy'
)

```
Since in fashion_mnist, every value in dataset has value b/w 0 & 255, it is good to **Normalize** data to make values b/w 0 and 1, by diving whole array by 255.

```python
import matplitlib.pyplot as plt
pls.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

test_images, training_images = test_images/255.0, training_images/255.0

```

```python
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
```

Callback to end the epochs when it reaches a certain threshold after which accuracy remains constant.


```python
class LocalCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print('\nLoss is low so cancelling training')
            self.model.stop_training = True

model.fit(training_images, training_labels, epochs=5, callbacks=[LocalCallback()])

```


### References:
1. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb
2. 