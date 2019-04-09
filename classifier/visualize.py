import matplotlib.pyplot as plt

def draw_chart(x, y, title=None, x_labels='x-axis', y_labels='y-axis'):
    plt.plot(x, y)
    if title is not None:
        plt.title(title)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.ylim(bottom=0, top=max(y) * 2)    
    plt.show()