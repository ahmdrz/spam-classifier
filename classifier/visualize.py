import matplotlib.pyplot as plt
from .pretty_matrix import pretty_plot_confusion_matrix
from pandas import DataFrame


def draw_chart(x, y_list, title=None, x_label='x-axis', y_label='y-axis', y_labels=None):
    for i, y in enumerate(y_list):
        label = 'Label {}'.format(i) if y_labels is None else y_labels[i]
        plt.plot(x, y, label=label)

    if title is not None:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(bottom=0.5, top=1.1)
    plt.legend()
    plt.show()


def draw_matrix(data):
    data = DataFrame(data)
    pretty_plot_confusion_matrix(data)
