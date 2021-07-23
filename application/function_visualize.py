import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_line(data, T, types):
    x = np.arange(0,T+1)
    mylabel = types
    plt.figure(1)
    plt.subplot(111)
    
    for i in range(len(types)):
        plt.plot(x, data[:, i], label=mylabel[i])
    
    plt.legend()
    plt.title('Proportions of Types over Time')
    plt.grid(True)
    plt.ylim([-0.02,1.02])
    
    plt.show()

    plt.savefig('simdata.png',dpi=600,facecolor='w',edgecolor='w',
        orientation='portrait', papertype=None,format=None,transparent=False,
        bbox_inches=None,pad_inches=0.01,frameon=None)

def plot_stacked_area(data, T, types):
    x = np.arange(0,T+1)
    y = []
    mylabel = []
    for i in range(len(types)):
        y.append(data[:, i])
        mylabel.append(types[i])

    plt.stackplot(x, y, labels=mylabel)
    plt.legend(loc='upper right')
    plt.margins(0, 0)
    plt.title('stacked area chart')
    plt.show()