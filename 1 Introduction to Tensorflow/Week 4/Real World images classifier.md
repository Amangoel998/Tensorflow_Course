## Image Generator in Tensorflow
- We can point a directory and it will generate labels based on sub-directory under it automatically.
- First make sub-dirs of training and testing data and later make same directory structure under them.
- The code to define the neural network, train it with on-disk images, and then predict values for new images.
  
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)
```
1. Rescale to normalize the data
2. Point to sub-dir ie. training data or testing data instead of top most directory.
3. Now the images are normalized to 300x300 size images as they are loaded not when they are being pre-processed.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
1. There are 3 sets of convolutional layers.
2. There are 3 bytes for color per pixel for each image.
3. In output layer there's 1 neuron for 2 classes. Signifying binary classification (Here, classify image of Human vs Horse).
4. It is more efficient than using 2 neurons for softmax activation function.

```python
from tensorflow.keras.optimizers import RMSprop
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
1. With RMSprop we can adjust the learning rate to experiment with performance.
2. Loss function is binary_crossentropy instead of classifier_crossentropy.

```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=2
)
```

1. validation_generator is created similarly to train_generator but in different object.
2. There are 1024 images in directory, so we're loading 128 at a time, hence we load it in 8 batches.

## References
1. https://www.youtube.com/watch?v=zLRB4oupj6g&feature=youtu.be
2. https://gombru.github.io/2018/05/23/cross_entropy_loss/
3. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
4. https://www.youtube.com/watch?v=eqEc66RFY0I&t=6s
5. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb
6. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb
7. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb