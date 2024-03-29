from typing import List, Optional, Tuple

import copy
import numpy as np
from numpy import ndarray

from RPP.utils.utils import (
    beautify_label,
    target_extractor,
    get_rates,
    curve_config_dict,
)
from RPP.utils.data import Cutter


class Model:
    def __init__(
        self,
        model_name: str,
        database: str,
        predictions: ndarray,
        truths: ndarray,
        event_nos: ndarray,
        original_truths: ndarray,
        energy: ndarray,
        lepton_pos: ndarray,
        color: str,
        cut_functions: Optional[List[Cutter]] = None,
        pulsemap_name: Optional[str] = None,
    ):
        self._name = model_name
        self._predictions = predictions
        self._truths = truths
        self._event_nos = event_nos
        self._original_truths = original_truths
        self._energy = energy
        self._lepton_pos = lepton_pos
        self._color = color
        self._cut_functions = copy.deepcopy(cut_functions)

        self._benchmark_index = None

        self._label = beautify_label(model_name)[1:-1]
        db_name = database.split("/")[-1][:-3]

        self._db_path = database
        self._title = db_name + "_" + model_name
        self._db_name = db_name
        self._pulsemap_name = (
            db_name.split("_")[0] if pulsemap_name is None else pulsemap_name
        )

    def add_benchmark(self, index: int):
        self._benchmark_index = index


class ClassificationModel(Model):
    def __init__(
        self,
        model_name: str,
        database: str,
        predictions: ndarray,
        truths: ndarray,
        event_nos: ndarray,
        original_truths: ndarray,
        energy: ndarray,
        lepton_pos: ndarray,
        color: str,
        cut_functions: Optional[List[Cutter]] = None,
        pulsemap_name: Optional[str] = None,
        target_rates: Optional[List[List[float]]] = None,
        target_cuts: Optional[List[List[float]]] = None,
        target_curve_type: Optional[str] = "ROC",
        reverse: Optional[bool] = False,
    ):
        super().__init__(
            model_name,
            database,
            predictions,
            truths,
            event_nos,
            original_truths,
            energy,
            lepton_pos,
            color,
            cut_functions,
            pulsemap_name,
        )

        # Get target rates based on what is specified
        target_rates, bg_rates = target_extractor(target_rates)
        target_cuts, bg_cuts = target_extractor(target_cuts)

        self._target_rates = target_rates
        self._bg_rates = bg_rates
        self._target_cuts = target_cuts
        self._bg_cuts = bg_cuts
        self._target_curve_type = target_curve_type

        self.flush_cut_functions()

        if reverse:
            self.invert_results()

        self._background_model = None
        self._performance_curves = {"ROC": None, "PRC": None}
        self._performance_rates = {"ROC": None, "PRC": None}

    def invert_results(self):
        self._predictions = 1 - self._predictions
        self._truths = 1 - self._truths
        self._original_truths = 1 - self._original_truths

    def get_background_model(self) -> Model:
        # Check if a background model already exists
        if self._background_model is None:
            background_model = copy.deepcopy(self)
            background_model._name = self._name + "_BG"
            background_model._label = beautify_label(background_model._name)[1:-1]
            background_model.invert_results()
            background_model._performance_curves = {"ROC": None, "PRC": None}
            background_model.flush_cut_functions()

            background_model._target_rates = self._bg_rates
            background_model._bg_rates = self._target_rates
            background_model._target_cuts = self._bg_cuts
            background_model._bg_cuts = self._target_cuts

            if self._performance_rates[self._target_curve_type] is not None:
                background_model.calculate_target_rates()

            self._background_model = background_model

        return self._background_model

    def get_performance_curve(self, curve_type: str) -> Tuple[float]:
        # Check of the model already has information for the desired curve
        if self._performance_curves[curve_type] is None:
            # Calculate curve parameters and add to dictionary
            curve_config = curve_config_dict[curve_type]
            x_rate, y_rate, thresholds = curve_config['metric_function'](
                self._truths, self._predictions
            )
            auc = curve_config['metric_score'](self._truths, self._predictions)

            self._performance_curves[curve_type] = (x_rate, y_rate, thresholds, auc)

        return self._performance_curves[curve_type]

    def calculate_target_rates(self, curve_type: Optional[str] = None):
        curve_type = self._target_curve_type if curve_type is None else curve_type

        # Get performance curve parameters and calculate rates for the desired curve type
        x_rate, y_rate, thresholds, _ = self.get_performance_curve(curve_type)
        self._performance_rates[curve_type] = get_rates(
            x_rate,
            y_rate,
            thresholds,
            self._target_rates,
            self._target_cuts,
            curve_type,
        )

        # Calculate cutter rates if the curve type is the target type
        if curve_type == self._target_curve_type:
            # Add the the rate info to each cut function
            for cutter in self._cut_functions:
                if cutter._performance_rates is None:
                    cutter_rates = copy.deepcopy(
                        self._performance_rates[self._target_curve_type]
                    )[0]

                    if cutter._checkpoint:
                        ones_ratio = np.count_nonzero(
                            self._truths == 1
                        ) / np.count_nonzero(self._original_truths == 1)
                        zeros_ratio = np.count_nonzero(
                            self._truths == 0
                        ) / np.count_nonzero(self._original_truths == 0)

                        # Scale the required rates
                        if self._target_curve_type == "ROC":
                            cutter_rates[0] *= zeros_ratio
                            cutter_rates[1] *= ones_ratio
                        if self._target_curve_type == "PRC":
                            cutter_rates[0] *= ones_ratio

                    cutter._performance_rates = cutter_rates

    def flush_cut_functions(self):
        # Remove rate info to each cut function
        for cutter in self._cut_functions:
            cutter._performance_rates = None

    def get_performance_iterator(
        self, label: str, as_bg: Optional[bool] = False
    ) -> List[Cutter]:
        # Get threshold and construct name
        threshold = self._performance_rates[self._target_curve_type][0][2]
        if as_bg:
            threshold = 1 - threshold
        name = r"Cut$_{%s} = %.3f$" % (label, threshold)

        # Create placeholder cutter and concatenate with cut function list
        default_cutter = Cutter(name, checkpoint=True)
        default_cutter._performance_rates = self._performance_rates[
            self._target_curve_type
        ][0]

        return [default_cutter] + self._cut_functions
    
class RegressionModel(Model):
    def __init__(
            self,
            model_name: str,
            database: str,
            predictions: ndarray,
            truths: ndarray,
            event_nos: ndarray,
            original_truths: ndarray,
            energy: ndarray,
            lepton_pos: ndarray,
            color: str,
            cut_functions: Optional[List[Cutter]] = None,
            pulsemap_name: Optional[str] = None,
        ):
            super().__init__(
                model_name,
                database,
                predictions,
                truths,
                event_nos,
                original_truths,
                energy,
                lepton_pos,
                color,
                cut_functions,
                pulsemap_name,
            )