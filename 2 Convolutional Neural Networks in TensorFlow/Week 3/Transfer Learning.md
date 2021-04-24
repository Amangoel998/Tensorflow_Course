## Transfer Learning

- Instead of self-training and modelling DL neural networks, we can instead use open-source models that is pre-trained
  by huge dataset already and further train it using smaller dataset that is specific to us.
- We can download some models, set some models as trainable and lock already trained layers and run rest of the model
  for use.
- Transfer Learning can take an existing model, freeze many of its layers to prevent them being retrained, and
  effectively 'remember' the convolutions it was trained on to fit images.

- Pre-trained models have convolutional layers and they're here intact with features that have already been learned.
- So you can lock them instead of retraining them on your data, and have those just extract the features from your data
  using the convolutions that they've already learned.
- Then you can take a model that has been trained on a very large datasets and use the convolutions that it learned when
  classifying its data.

Eg - Inception, Imagenet

## How to load pre-trained model

- Snapshot of model after being trained, the parameters can be then loaded into skeleton of the model to turn it back to
  trained model.

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights_file)

```

1. Get desired input shape for the model.
2. InceptionV3 has fully connected layer at the top, so ignore that and get straight to convolutions.
3. After having pretrained model instantiated, it can iterate through its layers and lock them, saying that they're not
   going to be trainable with this code.

- All Layers have name, look last layer name using

```python
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
```

- Moved the layer description a little up to find mixed7 which is output of lot of convolutions that are 7 by 7.
- We will grab output form this layer in inception.

```python
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])
```

- Create a new model taking input from pre-trained and a new layer we just created, using Abstract Model class.
- We then use ImageDataGenerator as previous and add augmentation params to it.
- Then we get training data from generator that gets flow from directory.

## References

1. https://www.tensorflow.org/tutorials/images/transfer_learning
2. https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels
3. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb