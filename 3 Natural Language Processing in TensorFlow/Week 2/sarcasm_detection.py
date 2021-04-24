import json
import numpy as np

with open("sarcasm.json", "r") as f:
    datasore = json.load(f)

sentences = []
labels = []

for item in datasore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000  # 10000
embedded_dim = 16
max_length = 16  # 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# Tweaking vocab_size->1000, max_length->16 and embedding->32
# May dec. loss but may also decrease accuracy.
# Hence, these hyper-parameters should be experimented with to get best results.

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]

training_labels = np.array(labels[:training_size])
testing_labels = np.array(labels[training_size:])

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences,
    padding=padding_type,
    maxlen=max_length,
    truncating=trunc_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences,
    padding=padding_type,
    maxlen=max_length,
    truncating=trunc_type
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedded_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

num_epochs = 30
history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2
)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
