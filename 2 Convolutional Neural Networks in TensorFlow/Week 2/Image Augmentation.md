## Image Augmentation

- Image Augmentation is a very simple, but very powerful tool to help you avoid overfitting your data. The concept is
  very simple though: If you have limited data, then the chances of you having data to match potential future
  predictions is also limited, and logically, the less data you have, the less chance you have of getting accurate
  predictions for data that your model hasn't yet seen. To put it simply, if you are training a model to spot cats, and
  your model has never seen what a cat looks like when lying down, it might not recognize that in future.
- Augmentation simply amends your images on-the-fly while training using transforms like rotation. So, it could '
  simulate' an image of a cat lying down by rotating a 'standing' cat by 90 degrees. As such you get a cheap way of
  extending your dataset beyond what you have already.

Eg.- We used following to resclae on fly w/o affecting files on disk.

`train_datagen = ImageDataGenerator(rescale=1./255)`

> Note that it's referred to as preprocessing for a very powerful reason: that it doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset.

## Overfitting

- Suppose model seems to be very good at spotting something from a limited dataset, but getting confused when you see
  something that doesn't match your expectations.
- If your images are fed into the training with augmentation such as a rotation, the feature might then be spotted, even
  if you don't have a cat reclining, your upright cat when rotated, could end up looking the same.

```python
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,  # 0-180
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

1. Image will rescale to fit 0-1 float value
2. Will rotate the image randomly b/w 0-40 degrees.
3. Shifting moves the image around inside of the frame. Here - 20% vertically & horizontally.
4. Shearing will skew the image along x-axis. Here by 20%.
5. Zooms the image along the center of image. Here by 20%.
6. Will flip image horizontally.
7. Fill any pixels lost by the operations keeping uniformity by filling the nearest pixel value.
8. The fill_mode parameter attempts to recreate lost information after a transformation like a shear

## References

1. https://github.com/keras-team/keras-preprocessing
2. https://keras.io/api/preprocessing/image/
3. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb