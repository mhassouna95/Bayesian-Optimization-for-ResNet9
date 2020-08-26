import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.optimizers import SGD
from scipy.optimize import minimize
from scipy.stats import norm
from plotting_funcs import plot_approximation, plot_acquisition
from resnet_model import ResNet9, ResNetBlock, ConvBatchNormReLU
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel)


def create_model(learning_rate):
    """
    Creates a ResNet9 Model with SGD optimizer and given learning rate.
    Arguments:
    learning_rate -- learning rate hyperparameter.

    Returns:
    model -- a Model() instance in Keras.
    """
    model = ResNet9(input_shape=(28, 28, 1), classes=10, seed=None)
    sgd = SGD(lr=learning_rate)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def blackbox(x, y, learning_rate):
    """
    Calculates blackbox function for bayesian optimization
    by training a ResNet9 model.

    Arguments:
    x -- image data.
    y -- labels.
    learning_rate -- learning rate hyperparameter.

    Returns:
    validation accuracy of the trained model.
    """
    batch_size = 128
    epochs = 3
    # create ResNet9 model
    model = create_model(learning_rate=learning_rate)

    # fit the model
    blackbox = model.fit(x=x,
                         y=y,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_split=0.15,
                         verbose=0
                         )
    # return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_accuracy'][-1]

    # print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # delete the Keras model with these hyper-parameters from memory.
    del model

    # clear the keras session
    K.clear_session()

    return accuracy


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Arguments:
        X -- Points at which EI shall be computed.
        X_sample -- Sample locations.
        Y_sample -- Sample values.
        gpr -- A GaussianProcessRegressor fitted to samples.
        xi -- Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''

    # predict mean and variance for all points
    mu, sigma = gpr.predict(X, return_std=True)

    sigma = sigma.reshape(-1, 1)

    # calculate the best sample (highest accuracy)
    sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def sample_next_hyperparameter(
        acquisition,
        X_sample,
        Y_sample,
        gpr,
        bounds,
        n_restarts=25):
    '''
    Proposes the next hyperparameter point by optimizing
    the acquisition function.

    Args:
        acquisition -- Acquisition function.
        X_sample -- Sample locations.
        Y_sample -- Sample values.
        gpr -- A GaussianProcessRegressor fitted to samples.
        bounds -- Bounds of the hyperparameter.
        n_restarts -- Number of restarts for finding the optimum.

    Returns:
        Location of the acquisition function maximum.
    '''

    dim = X_sample.shape[1]
    min_val = float('inf')
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


if __name__ == "__main__":
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # Load the data
    x_train = np.load(
        './data/kmnist-train-imgs.npz')['arr_0']
    x_test = np.load(
        './data/kmnist-test-imgs.npz')['arr_0']
    y_train = np.load(
        './data/kmnist-train-labels.npz')['arr_0']
    y_test = np.load(
        './data/kmnist-test-labels.npz')['arr_0']

    # plot random image of each class
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(
        num_row, num_col, figsize=(
            1.5 * num_col, 2 * num_row))
    for i in range(num_classes):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(x_train[y_train == i][np.random.randint(
            len(x_train[y_train == i]))], cmap='gray')
        ax.set_title('Label: {}'.format(i))
    plt.tight_layout()
    plt.draw()

    # reshape
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('{} train samples, {} test samples'
          .format(len(x_train), len(x_test)))

    # encode labels as one-hot
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # bounds of the learning rate parameter
    bounds = np.array([[1e-5, 1]])
    # define initial samples for gaussian process
    X_init = np.array([[1e-3], [2e-1]])
    Y_init = np.vstack((blackbox(x_train, y_train, X_init[0][0]),
                        blackbox(x_train, y_train, X_init[1][0])))
    Y_init = np.array(Y_init).reshape(-1, 1)

    # dense grid of points within bounds
    X = np.linspace(bounds[:, 0], bounds[:, 1], 10000).reshape(-1, 1)
    plt.plot(X_init, Y_init, 'kx', mew=5, label='Initial samples')
    plt.legend()

    # define gaussian process
    gpr_kernel = ConstantKernel(1.0) * Matern(length_scale=1, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=gpr_kernel)
    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    # Number of iterations
    n_iter = 10

    plt.figure(figsize=(24, n_iter * 6))
    plt.subplots_adjust(hspace=0.5)

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function
        # (expected_improvement)
        X_next = sample_next_hyperparameter(
            expected_improvement, X_sample, Y_sample, gpr, bounds)

        # Obtain next noisy sample from the objective function
        Y_next = blackbox(x_train, y_train, X_next[0][0])

        # Plot samples, surrogate function, noise-free objective
        # and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(
            gpr,
            X,
            X_sample,
            Y_sample,
            X_next,
            show_legend=i == 0)
        plt.title('Iteration {}'.format(i + 1))

        plt.subplot(n_iter, 2, 2 * i + 2)
        ei = expected_improvement(X, X_sample, Y_sample, gpr)
        plot_acquisition(X, ei, X_next, show_legend=i == 0)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    # end of bayesian opt process
    print("Sampled learning rates: ", np.hstack((X_sample, Y_sample)))
    best_learning_rate = X_sample[Y_sample.argmax()][0]
    print("Best learning rate: ", best_learning_rate)

    # train a model with the chosen learning rate
    model = create_model(best_learning_rate)
    batch_size = 128
    epochs = 3
    model.fit(x_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size,
              )

    # print train and test scores
    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', train_score[1])
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])
    plt.show()
