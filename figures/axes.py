from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from RPP.plotters.plotter import Plotter


class RPPAxes(Axes):

    def __init__(
        self, 
        fig: Figure, 
        mpl_axis: Axes, 
        plotter: Plotter,
        index: Union[int, List[int]],
        *args, 
        **kwargs,
    ):

        # Initialize axes
        super().__init__(
            mpl_axis.get_figure(), 
            mpl_axis._position,
            facecolor = mpl_axis._facecolor,
            frameon = mpl_axis._frameon,
            sharex = mpl_axis._sharex,
            sharey = mpl_axis._sharey,
            label = mpl_axis.get_label(),
            xscale = mpl_axis.get_xscale(),
            yscale = mpl_axis.get_yscale(),
            box_aspect = mpl_axis.get_box_aspect(),
        )

        self._plotter: Plotter = plotter
        self._figure_index = index
        self._subplot_args = args

        # Remove mpl axes and add new axes to figure
        fig.delaxes(mpl_axis)
        projection_class, pkw = fig._process_projection_requirements(
            *args, **kwargs)
        key = (projection_class, pkw)
        fig._add_axes_internal(self, key)

    def shift_axis(
        self,
        a: List[float],
        b: List[float],
        shift_x: Optional[bool] = False,
        shift_y: Optional[bool] = False,
    ) -> Tuple[float]:
        
        # Shift axis if the dataset has outliers
        ax_max = max([np.percentile(a, 99.9), np.percentile(b, 99.9)])
        ax_min = min([np.percentile(a, 0.1), np.percentile(b, 0.1)])
        ax_mean = (np.mean(a) + np.mean(b)) / 2

        # Calculate shift
        dist = max([ax_max - ax_mean, ax_mean - ax_min])
        ax_min, ax_max = (
            (0, 1)
            if abs(1 - max(a)) < 1e-5
            else (ax_mean - dist * 1.1, ax_mean + dist * 1.1)
        )

        # Apply to the desired axisß
        if shift_x:
            self.set_xlim(ax_min, ax_max)
        if shift_y:
            self.set_ylim(ax_min, ax_max)

        return ax_min, ax_max

    def get_axis_neighbour(self, direction: str) -> Axes:

        assert (
            len(self._subplot_args) > 1 or
            (len(self._subplot_args) == 1 and self._subplot_args[0] != 1) 
        ), "Cannot find neighbour on figure with only 1 subplot."

        dir_dict = {'above': -1, 'below': 1, 'left': -1, 'right': 1}
        self_index = self._figure_index

        if direction in ['left', 'right']:
            neighbour_index = self_index + dir_dict[direction]

        elif direction in ['above', 'below']:
            if len(self._subplot_args) == 1:
                neighbour_index = self_index + dir_dict[direction]
            elif len(self._subplot_args) > 1:
                neighbour_index = self_index + dir_dict[direction]*self._subplot_args[1]
        
        else:
            print("Please specify directon as 'above', 'below', 'left', or 'right'.")
            exit()

        return self.get_figure().get_axes()[neighbour_index]
