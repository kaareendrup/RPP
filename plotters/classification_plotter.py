from typing import Dict, List, Optional

from RPP.plotters.plotter import Plotter
from RPP.data.models import ClassificationModel
from RPP.figures.classification_axes import ClassificationAxes
from RPP.utils.utils import beautify_label
from RPP.utils.style import basic_pos_dict


class ClassificationPlotter(Plotter):
    def __init__(
        self,
        name: str,
        plot_dir: str,
        target: str,
        background: str,
        pos_dict: Dict[bool, Dict[bool, float]] = basic_pos_dict,
        show_cuts: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(name, plot_dir, target, **kwargs)

        # Add classification specific parameters
        self._model_class = ClassificationModel
        self._axes_class = ClassificationAxes
        self._background = background
        self._background_label = beautify_label(background)
        self._show_cuts = show_cuts
        self._pos_dict = pos_dict
