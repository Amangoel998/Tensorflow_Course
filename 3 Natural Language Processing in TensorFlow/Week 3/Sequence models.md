## Sequence Models
- We tokenized the words but aour classifier failed to get meaningful results.
- The context of words were hard to follow.
- The sub-words and the sequence they follow becomes important to understand the whole meaning of a sentence.
- Since Ml takes data and labels to give out rules to predict new results, it doesn't take into account the sequence in which it is given.

# Fibonacci of numbers
- As an example, let's take fibonacci numbers sequence and replace each value in series with a variable try to derive an equation.
- It gives, An = An-1 + An-2
- Thus, y = func(x, y-1) or y = func(x, func(x-1))
- Hence there is a horizontal connection to each NN layer output to the other.

- Recurrent Neural Network(RNN) are a type of Neural Network where the output from previous step are fed as input to the current step.
- The main and most important feature of RNN is Hidden state, which remembers some information about a sequence.
- Formula for activation --> y = tanh(Wr * y-1 + Wi * x)
- Wr = Weight of recurrent neuron
- Wi = Weight of input neuron.

- Formula for output --> o = Wo * y
- Wo = Weight of output layer

# Long short - Term memory
- Along with context passed by RNN, LSTM is has additional cell state pipeline which can pass through the network to impact it.
- This helps keep context from earlier tokens relevance in later ones, to avoid situation like guessing irish dur to word Ireland in sentence, but actual language being Gaelic.
- Values from earlier words can be carried to later ones via a cell state

- Cell State ca also be bi-directional.
- So later context can impact earlier ones.

## LSTM in code
What is the purpose of the embedding dimension?
It is the number of dimensions for the vector representing the word encoding
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- Bidirectional doubles the input size resulting output shape to (None, 128) from that layer.
- To feed LSTM to next layer, pass return_sequences=True in LSTM layers.
- The model with 2 layer LSTM has lesser & lower dips.
- Same is true for loss as well, since validation loss increased overtime for both, the spikes were more sharper in 1 layer LSTM model.

- Jaggedness in model output graph, indicates that model needs further improvement.

## References
1. https://www.coursera.org/lecture/nlp-sequence-models/deep-rnns-ehs0S
2. https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/
3. https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay
4. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201b.ipynb
5. For Sentiment Analysis Dataset - https://www.kaggle.com/kazanova/sentiment140