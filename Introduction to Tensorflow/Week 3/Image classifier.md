## Convolution
    Well, if you've ever done any kind of image processing, it usually involves having a filter and passing that filter over the image in order to change the underlying image. 
    For every pixel, take its value, and take a look at the value of its neighbors.
    Then to get the new value for the pixel, we simply multiply each neighbor by the corresponding value in the filter.

    Eg.
    0    64    128
    48   192   144
    142  226   168
    Current_Pixel_value = 192
    -1    0    -2
    .5   4.5  -1.5
    1.5   2    -3
    New_Pixel_Value = (0*-1) + (0*64) + (-2*128) +
                (.5*48) + (4.5*192) + (-1.5*144) +
                (1.5*142) + (2*226) + (-3*168)
            = 577
    
    eg.
    -1 -2 -1                                    -1  0  1
    0   0  0                                    -2  0  2
    1   2  1                                    -1  0  1
    Reveals Horizontal lines of an image        Reveals Vertical Lines of an image

## Pooling
    It is a way of compressing images
    Eg. - For each four pixels, pick the largest value and re-create an image.

## Using Tensorflow to Implement convolutional layers
    ```python

    ```
    ```python
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ```
    1. keras will generate 64 filters for us, these filters are 3x3, with relu as activation func to throw out negative values. The input shape has extra one to tally for color depth using 1 byte.
    2. Max Pooling to pool images and get maximum values. For 3x3 ie. 9 block pool, the max value will be used.
    3. Another convolutional and pooling layer to learn another set on top of existing one,
    4. Since filter is 2x2, the image will be 2 less from top and below, hence giving 26x26 since calculation cannot be performed on extreme edges of image.
    5. The output of conv2d layer on 28x28 image, after 3x3 filter, is 26x26, since in a 9x9 matrix the calculation cannot be done on top, bottom, left and right edge of image.
    6. After passing a 3x3 filter over a 28x28 image, the output image will be 26x26 pixels big.
    7. After max pooling a 26x26 image with a 2x2 filter, the output will be 13x13 big.

    You’ve now seen how to turn your Deep Neural Network into a Convolutional Neural Network by adding convolutional layers on top, and having the network train against the results of the convolutions instead of the raw pixels.
    

    cv2 library in python,
    misc lib from scipy, use misc.ascent() return an image 

## Reference:
1. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb
2. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=9FGsHhv6JvDx
3. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb 
4. https://lodev.org/cgtutor/filtering.html