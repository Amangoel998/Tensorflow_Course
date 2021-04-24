## Dropouts

- Even after augmentation and convolutional layers we are still facing over-fitting problem.
- Layers in NN can sometimes have similar weights and possibly impact each other.
- Hence, using dropouts, we can drop some neurons and take advantage of back-propagation which helps increasing weight
  impact.

```python
x = layers.Dropout(0.2)(x)
```

- Dropping 20% of neurons.

## References

1. https://www.youtube.com/watch?v=ARq74QuavAo
2. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb