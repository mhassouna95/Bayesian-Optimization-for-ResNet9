import numpy as np
import matplotlib.pyplot as plt


def plot_approximation(gpr, X, X_sample, Y_sample, X_next=None, show_legend=False):
    """
    Plots the posterior mean and variance of the surrogate model as well as the observed samples.

    Arguments:
    gpr -- Surrogate model, a GaussianProcessRegressor.
    X -- range of X values for the plot.
    X_sample -- values of the hyperparameter samples.
    Y_sample -- values of the blackbox function at X_sample.
    X_next -- boolean, controls the display of the position of the selected hyperparameter.
    show_legend -- boolean, controls the display of the legend.
    Returns:
    model -- a Model() instance in Keras.
    """

    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def plot_acquisition(X, Y, X_next, show_legend=False):
    """
    Plots the acquisition function.

    Arguments:
    X -- range of X values for the plot.
    Y -- values of the acquisition function at points X.
    X_next -- boolean, controls the display of the position of the selected hyperparameter.
    show_legend -- boolean, controls the display of the legend.
    Returns:
    model -- a Model() instance in Keras.
    """
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()
