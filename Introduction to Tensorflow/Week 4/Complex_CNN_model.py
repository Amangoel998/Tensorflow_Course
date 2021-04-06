import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

DESIRED_ACCURACY = 0.999
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > DESIRED_ACCURACY:
            print('\nReached 99.9% accuracy so cancelling training!')
            self.model.stop_training = True


# This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
loss='binary_crossentropy',
metrics=['accuracy'])
    

# This code block should create an instance of an ImageDataGenerator called train_datagen 
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

# Please use a target_size of 150 X 150.
train_generator = train_datagen.flow_from_directory(
    '/tmp/h-or-s',
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)
# Expected output: 'Found 80 images belonging to 2 classes'

# This code block should call model.fit_generator and train for a number of epochs.
# model fitting
history = model.fit_generator(
    train_generator,
    steps_per_epoch=2,
    epochs=20,
    verbose=2,
    callbacks=[myCallback()])
# model fitting