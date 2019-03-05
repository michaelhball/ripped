import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


__all__ = ['plot_train_test_loss']


def plot_train_test_loss(train_losses, test_losses, display=True, save_file=None):
    """
    Plots loss through training for train and test datasets
    """
    fig, ax = plt.subplots()
    ax.plot(train_losses, 'b-', label='train loss')
    ax.plot(test_losses, 'r-', label='test loss')
    ax.legend()
    ax.set(xlabel='epoch number', ylabel='average loss', title='train/test losses')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    if save_file:
        plt.savefig(save_file)
    if display:
        plt.show()
