from keras.models import Sequential, Model
from keras.initializers import GlorotUniform
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D)


def ConvBatchNormReLU(
        input,
        filters,
        kernel_size,
        name,
        stride,
        padding="SAME",
        seed=None):
    """
    Implementation of a convolution followed by a BatchNormalization
    and a ReLU layer.

    Arguments:
    input -- input tensor.
    filters -- list of integers, defines number of filters in the CONV layers.
    kernel_size -- integer, size of the convolution filter.
    name -- integer, name of layers depending on their position.
    stride -- integer, value for the stride of the conv layer.
    padding -- string/character, padding of the conv layer.
    seed -- integer, random seed for initialization.

    Returns:
    X -- output tensor after Conv -> BatchNorm -> ReLU.
    """
    k = (kernel_size, kernel_size)
    s = (stride, stride)
    X = Conv2D(filters, k, padding=padding, strides=s, name='conv' +
               str(name), kernel_initializer=GlorotUniform(seed=seed))(input)
    X = BatchNormalization(axis=3, name='bn_conv' + str(name))(X)
    X = Activation('relu', name='relu' + str(name))(X)

    return X


def ResNetBlock(
        input,
        filters,
        kernel_size,
        name,
        stride,
        padding="SAME",
        pooling=2,
        seed=None):
    """
    Implementation of a convolution followed by a BatchNormalization
    and a ReLU layer.

    Arguments:
    input -- input tensor.
    filters -- list of integers, defines number of filters in the CONV layers.
    kernel_size -- integer, size of the convolution filter.
    name -- integer, name of layers depending on their position.
    stride -- integer, value for the stride of the conv layer.
    padding -- string/character, padding of the conv layer.
    pooling -- integer, filter size of the MaxPool layer.
    seed -- integer, random seed for initialization.

    Returns:
    X -- output tensor after a ResNet basic block.
    """

    X = ConvBatchNormReLU(
        input,
        filters,
        kernel_size,
        str(name) + "a",
        stride,
        padding,
        seed)
    X = MaxPooling2D((pooling, pooling), name="max_pool" + str(name))(X)
    Y = ConvBatchNormReLU(
        X,
        filters,
        kernel_size,
        str(name) + "b",
        stride,
        padding,
        seed)
    Y = ConvBatchNormReLU(
        Y,
        filters,
        kernel_size,
        str(name) + "c",
        stride,
        padding,
        seed)
    X = Add(name="add" + str(name))([X, Y])

    return X


def ResNet9(input_shape=(28, 28, 1), classes=10, seed=None):
    """
    Implementation of the ResNet9 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> ResNetBlock -> CONV2D -> BATCHNORM ->
    RELU -> MAXPOOL -> ResNetBlock -> MAXPOOL -> Dense.
    Arguments:
    input_shape -- shape of the images of the dataset.
    classes -- integer, number of classes.
    seed -- integer, random seed for initialization.

    Returns:
    model -- a Model() instance in Keras.
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # prep
    X = ConvBatchNormReLU(
        X_input,
        filters=64,
        kernel_size=3,
        name=0,
        stride=1,
        padding="SAME",
        seed=seed)

    # layer1
    X = ResNetBlock(
        X,
        filters=128,
        kernel_size=3,
        name=1,
        stride=1,
        padding="SAME",
        pooling=2,
        seed=seed)

    # layer2
    X = ConvBatchNormReLU(
        X,
        filters=256,
        kernel_size=3,
        name=2,
        stride=1,
        padding="SAME",
        seed=seed)
    X = MaxPooling2D((2, 2), name="max_pool" + str(2))(X)

    # layer3
    X = ResNetBlock(
        X,
        filters=512,
        kernel_size=3,
        name=3,
        stride=1,
        padding="SAME",
        pooling=2,
        seed=seed)

    # classifier
    X = MaxPooling2D((2, 2), name="max_pool_output")(X)
    X = Flatten()(X)
    X = Dense(
        classes,
        activation='softmax',
        name='fc' +
        str(classes),
        kernel_initializer=GlorotUniform(seed=seed))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet9')

    return model
