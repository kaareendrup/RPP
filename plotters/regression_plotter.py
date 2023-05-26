from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from RPP.plotters.plotter import Plotter
from RPP.data.models import Model
from RPP.data.models import RegressionModel
from RPP.figures.regression_axes import RegressionAxes
from RPP.utils.maths.maths import bin_residual_width, w_errorprop


class RegressionPlotter(Plotter):
    def __init__(self, name: str, plot_dir: str, target: str, unit: str, **kwargs):
        super().__init__(name, plot_dir, target, **kwargs)

        self._unit = unit
        self._unit_str = " [" + self._unit + "]"

        self._bin_positions = []
        self._bin_centers = []
        self._n_residual_bins = []

        self._model_class = RegressionModel
        self._axes_class = RegressionAxes
