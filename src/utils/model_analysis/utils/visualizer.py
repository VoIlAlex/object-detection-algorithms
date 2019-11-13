import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.ion()

import numpy as np
import time


__all__ = ['Visualizer']


class Func:
    def __init__(self, axes, color='b', name=None):
        self.x = np.array([])
        self.y = np.array([])
        self.color = color
        self.axes = axes
        self.curve = self.axes.plot(self.x, self.y, self.color, label=name)[0]
        self.axes.legend(loc='best')

    def update(self, new_x, new_y):
        self.x = np.append(self.x, new_x)
        self.y = np.append(self.y, new_y)
        self.curve.set_xdata(self.x)
        self.curve.set_ydata(self.y)
        self._rescale()

    def _rescale(self):
        self.axes.relim()
        self.axes.autoscale_view()


class Visualizer:
    def __init__(self, curves_descriptors: list, x_label=None, y_label=None):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(1, 1, 1)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.autoscale(enable=True, axis='both')
        self.funcs = {d['name']: Func(
            self.axes, d['color'], d['name']) for d in curves_descriptors}

    def update(self, new_values: dict):
        for func_name in new_values:
            func = self.funcs[func_name]
            values = new_values[func_name]
            func.update(values['x'], values['y'])

    def redraw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == '__main__':
    funcs_descriptors = [
        {'name': 'train', 'color': 'r'},
        {'name': 'test', 'color': 'b'},
    ]
    visualizer = Visualizer(funcs_descriptors, 'x_label', 'y_label')

    def rand(): return np.random.uniform(0, 10)

    for i in range(10):
        visualizer.update({
            'train': {'x': rand(), 'y': rand()},
            'test': {'x': rand(), 'y': rand()},
        })
        visualizer.redraw()
        time.sleep(1)
