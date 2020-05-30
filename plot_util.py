import os
from time import strftime, gmtime

import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional

from matplotlib.axes import Axes

orange = '#FF8000'


class PlotWrapper:
    def __init__(self, lower, upper, logs: bool = False):
        self.upper = upper
        self.lower = lower
        self.funcs: List[Callable[[Axes], None]] = list()
        self.fig = plt.figure()
        self.ax: Axes = plt.gca()
        self.legends: List[str] = list()
        self.logs = logs

    def run(self):
        for f in self.funcs:
            f(self.ax)
        self.ax.legend(tuple(self.legends), loc='upper right')
        self.ax.grid(True)

    def show(self):
        self.run()

        plt.show()
        plt.close(self.fig)

    def saveToFile(self, fileName: str):
        self.run()

        time = strftime("_%d-%m-%H:%M:%S", gmtime())
        plt.savefig(os.path.join('graphics', (fileName + time + '.png')), dpi=200)
        plt.close(self.fig)

    def add_text(self, text):
        plt.text(0.1, 0.1, text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=self.ax.transAxes,
                 fontsize='xx-large',
                 bbox={'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.8})

    def plot(self, xs, ys, color):
        if self.logs:
            plt.semilogx(xs, ys, color)
        else:
            plt.plot(xs, ys, color)

    def add_line(self, xs, ys, legend: str, color: Optional[str] = None):
        self.funcs.append(lambda _: self.plot(xs, ys, color))
        self.legends.append(legend)

    def horizontal_line(self, value, legend: str, color: Optional[str] = None):
        self.funcs.append(lambda _: self.ax.hlines(value, self.lower, self.upper, color))
        self.legends.append(legend)

    def vertical_line(self, value, legend: str, color: Optional[str] = None):
        self.funcs.append(lambda _: self.ax.vlines(value))
        self.legends.append(legend)

    def x_label(self, label: str):
        self.funcs.append(lambda _: plt.xlabel(label, fontsize='xx-large'))

    def y_label(self, label: str):
        self.funcs.append(lambda _: plt.ylabel(label, fontsize='xx-large'))
