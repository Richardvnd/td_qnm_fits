import matplotlib.pyplot as plt
import numpy as np


class PlotConfig:
    def __init__(self):
        self.style = "stylesheet.mplstyle"
        self.fig_width = 246.0 * (1.0 / 72.27)
        self.fig_width_2 = 510.0 * (1.0 / 72.27)
        self.fig_height = self.fig_width / 1.618
        self.fig_height_2 = self.fig_width_2 / 1.618
        self.colormap = plt.cm.viridis
        COLOR_TOP = 0.95
        self.colors = [self.colormap(COLOR_TOP * x**0.5) for x in np.linspace(0, 1, 8)]
        self.colormap2 = plt.cm.magma
        self.color_bounce = "#991946"
        self.color_lr = "#c47a00"
        self.color_td = plt.cm.plasma(0.15)
        self.color_const = plt.cm.plasma(0.60)

    def colors2(self, n):
        """``n`` magma colours, trimmed of the extremes to stay visible on white."""
        return [self.colormap2(x) for x in np.linspace(0.1, 0.85, n)]

    def apply_style(self):
        plt.style.use(self.style)
