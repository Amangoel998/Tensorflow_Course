## Tokenizing
- Each word can be tokenized and given a value.
- Tokenize each words to form a dictionary of words to make a corpus along with corresponding integer values.


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
]

tokenizer = Tokenizer(
    num_words=100, # max words to be tokenized, and picks the most common ‘n’ words
    oov_token='<OOV>' # Token in case of new word
)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print(word_index)
print(sequences)

test_data = [
    'I really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
```

## Special Values for unfamiliar words in test_data
- If we try to tokenize the words, in test_data, which are unfamiliar to our tokenizer, then it creates sentence sequences not having those words tokenized.
- We can also add a special value in place of those words which are not familiar.
- It is thus recommended to make training data of high vocabulary size.

## Padding
- Then we make sentences into list of values based on tokens.
- Also make sentences with atleast same length for training easier.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(
    sequences,
    padding='post' # For padding at end instead of front as default
    maxlen=5 # Max len of sequence, lose words from start of sentence
    truncating = 'post' # Allows losing words from ending of sentences
    )
```


## References
1. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%201.ipynb
2. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%202.ipynb#scrollTo=rX8mhOLljYeM
3. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%203.ipynb
4. http://mlg.ucd.ie/datasets/bbc.html
5. https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js