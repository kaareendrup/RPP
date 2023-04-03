from typing import Dict, List, Tuple, Optional, Callable

import pandas as pd
from matplotlib.colors import ListedColormap

from RPP.utils.fiTQun_schemes import pred_pure
from RPP.utils.data import query_database, Cutter
from RPP.data.models import Model
from RPP.utils.utils import make_plot_dir, fiTQun_dict, beautify_label
from RPP.utils.style import basic_colormap, basic_color_dict, basic_style_dict


class Plotter:
    def __init__(
        self,
        name: str,
        plot_dir: str,
        target: str,
        color_dict: Optional[Dict[str, List[str]]] = basic_color_dict,
        style_dict: Optional[Dict[str, Dict[str, str]]] = basic_style_dict,
        cmap: Optional[ListedColormap] = basic_colormap,
    ):
        # Member variables # TODO: Create common string that is prepended to plot names
        self._model_class = Model
        self._plot_dir = make_plot_dir(name, target, plot_dir)
        self._target = target
        self._cmap = cmap
        self._color_dict = color_dict
        self._style_dict = style_dict

        self._target_label = beautify_label(target)

        self._models_list: List[Model] = []
        self._benchmarks_list: List[Model] = []

    def load_csv(
        self,
        file: str,
        database: Optional[str] = None,
        cut_functions: Optional[List[Cutter]] = None,
        target: Optional[str] = None,
        **kwargs
    ):
        data = pd.read_csv(file)
        data.sort_values("event_no", inplace=True, ignore_index=True)
        event_nos = data["event_no"].to_numpy()

        if target is None:
            target = self._target

        original_truths = data[target].to_numpy()

        # Apply the relevant cuts
        if cut_functions is not None:
            if database is not None:
                for function in cut_functions:
                    event_nos = function.cut(event_nos, database)

                    # Replace original truths if the function is not a checkpoint
                    if not function._checkpoint:
                        original_truths = data[data["event_no"].isin(event_nos)][
                            target
                        ].to_numpy()

                data = data[data["event_no"].isin(event_nos)]
            else:
                print("No database specified, unable to apply cuts")

        preds, truths = data[target + "_pred"].to_numpy(), data[target].to_numpy()

        # Get energy and position data
        if database is not None:
            try:
                features_query = "SELECT fLE, fVx, fVy, fVz, event_no FROM truth WHERE event_no IN {}".format(
                    tuple(event_nos)
                )
                features = query_database(database, features_query)
                features.sort_values("event_no", inplace=True, ignore_index=True)
                energy = features["fLE"].to_numpy()
                lepton_pos = features[["fVx", "fVy", "fVz"]].to_numpy()
            except:
                print("No values extracted from database")
                energy = None
                lepton_pos = None
        else:
            print("No database specified, no energy and position extracted")
            energy = None
            lepton_pos = None

        return preds, truths, event_nos, energy, original_truths, lepton_pos

    def add_results(
        self,
        results_file: str,
        model_name: str,
        database_file: Optional[str] = None,
        color: Optional[str] = None,
        cut_functions: Optional[List[Cutter]] = None,
        **kwargs
    ):
        # Get color from dict if it is not defined
        if color is None:
            color = self._color_dict["model"][len(self._models_list)]

        # Load data from csv
        preds, truths, event_nos, energy, original_truths, lepton_pos = self.load_csv(
            results_file, database_file, cut_functions, **kwargs
        )

        # Define model parameters and append to list
        model = self._model_class(
            model_name,
            database_file,
            preds,
            truths,
            event_nos,
            original_truths,
            energy,
            lepton_pos,
            color,
            cut_functions,
            **kwargs,
        )

        self._models_list.append(model)

    def add_benchmark(
        self,
        benchmark_file: str,
        model_name: str,
        pred_scheme: Optional[Callable] = pred_pure,
        link_models: Optional[List[str]] = None,
        color: Optional[str] = None,
        database_file: Optional[str] = None,
        cut_functions: Optional[List[Cutter]] = None,
        **kwargs
    ):
        # Get color from dict if it is not defined
        if color is None:
            color = self._color_dict["benchmark"][len(self._benchmarks_list)]

        # Define model parameters and append to list
        if benchmark_file[-4:] == ".csv":
            # Load data
            (
                preds,
                truths,
                event_nos,
                energy,
                original_truths,
                lepton_pos,
            ) = self.load_csv(benchmark_file, database_file, cut_functions, **kwargs)

        elif benchmark_file[-3:] == ".db":
            # Use the event numbers and truth from the first linked model if supplied, else use first model in plotter
            if link_models is not None:
                truth_model = self.get_models_by_names(link_models, self._models_list)[
                    0
                ]
            else:
                truth_model = self._models_list[0]

            truths = truth_model._truths
            event_nos = truth_model._event_nos
            energy = truth_model._energy
            original_truths = truth_model._original_truths
            lepton_pos = truth_model._lepton_pos
            cut_functions = truth_model._cut_functions

            # # Get correct target for sqlite query
            model_map_name = model_name.split("_")[0]
            pred_target = (
                fiTQun_dict[self._target]
                if model_map_name == "fiTQun"
                else self._target
            )

            # # Get predictions from sqlite database
            preds = pred_scheme(benchmark_file, pred_target, model_map_name, event_nos)
            database_file = benchmark_file

        else:
            raise SystemExit("Incorrect file type. Please use .csv or .db")

        # Define model parameters and append to list
        model = self._model_class(
            model_name,
            database_file,
            preds,
            truths,
            event_nos,
            original_truths,
            energy,
            lepton_pos,
            color,
            cut_functions,
            **kwargs,
        )

        self._benchmarks_list.append(model)

        # Set as default benchmark for linked models
        if link_models is not None:
            for model in self._models_list:
                if model._name in link_models:
                    model.add_benchmark(len(self._benchmarks_list) - 1)

    def get_models_by_names(
        self, model_names: List[str], model_list: List[Model]
    ) -> List[Model]:
        result_list = []
        for name in model_names:
            for model in model_list:
                if model._name == name:
                    result_list.append(model)

        return result_list

    def get_models_and_benchmarks(
        self, model_names: List[str], benchmark_names: List[str]
    ) -> List[List[Model]]:
        # Get correct models and benchmarks if not supplied
        if model_names is None:
            models = self._models_list
        else:
            models = self.get_models_by_names(model_names, self._models_list)

        if benchmark_names is None:
            if len(self._benchmarks_list) == 0:
                benchmarks = [None] * len(models)
            else:
                benchmarks = [self._benchmarks_list[0]] * len(models)
        elif len(benchmark_names) == 1:
            benchmarks = [benchmarks] * len(models)
        else:
            benchmarks = self.get_models_by_names(
                benchmark_names, self._benchmarks_list
            )

        # Get benchmark if it is predefined
        for i in range(len(models)):
            if models[i]._benchmark_index is not None:
                benchmarks[i] = self._benchmarks_list[models[i]._benchmark_index]

        return [models, benchmarks]
