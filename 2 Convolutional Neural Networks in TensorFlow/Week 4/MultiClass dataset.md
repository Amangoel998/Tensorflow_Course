# Multi-Class Dataset
1. The class_mode is 'categorical' instead of 'binary'.
2. the output activation func is 'softmax' with all 3 values summing upto one.
3. Loss function is 'categorical_crossentropy' instead of 'binary_crossentropy'.

```python
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

```
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

```
```python
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['acc'])

```

**Note:** Since we are using image generators from directory, we get output in sorted order of their directory name ie Paper, Rock & Scissors.
## References
1. Dataset link - http://www.laurencemoroney.com/rock-paper-scissors-dataset/
2. Rock Paper & Scissors Model - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb
3. Kaggle dataset link - https://www.kaggle.com/datamunge/sign-language-mnist