import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def save_plot(training_loss, prec, val_loss, path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(training_loss, label='training loss')
    plt.plot(prec, label='validation precision')
    plt.plot(val_loss, label='validation loss')
    ax.legend()
    plt.savefig(path)
