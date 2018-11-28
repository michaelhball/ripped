import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


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


def plot_train_test_accs(train_accs, test_accs, display=True, save_file=None):
    """
    Plots accuracy through training for train and test datasets
    """
    fig, ax = plt.subplots()
    ax.plot(train_accs, 'b-', label='train accuracy')
    ax.plot(test_accs, 'r-', label='test accuracy')
    ax.set(xlabel='epoch number', ylabel='accuracy', title='train/test accuracies')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    if save_file:
        plt.savefig(save_file)
    if display:
        plt.show()
