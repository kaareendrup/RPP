from typing import Dict, List, Tuple, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from RPP.figures.axes import RPPAxes
from RPP.utils.utils import (
    calculate_alpha,
    curve_config_dict,
)

class ClassficationAxes(RPPAxes):

    def score_hist(
        self,
        model_names: Optional[List[str]] = None,
        n_bins: Optional[int] = 100,
        log_y: Optional[bool] = False,
        shift_x: Optional[bool] = False,
    ):

        # Add the correct model 
        models = self._plotter.get_models_by_names(model_names)

        for model, ls in zip(models, self._plotter._style_dict["histogram"]):
            
            # Subdivide in truth and background
            ones = model._predictions[np.where(model._truths == 1)]
            zeros = model._predictions[np.where(model._truths == 0)]

            if shift_x:
                xmin, xmax = self.shift_axis(ones, zeros, shift_x=True)
                bins = np.linspace(xmin, xmax, n_bins)
            else:
                bins = n_bins

            self.hist(
                ones,
                color=model._color,
                bins=bins,
                histtype="step",
                linestyle=ls,
                label=self._plotter._target_label + r" " + model._label,
                zorder=1,
            )
            self.hist(
                zeros,
                color="k",
                bins=bins,
                histtype="step",
                linestyle=ls,
                label=self._plotter._background_label + r" " + model._label,
                zorder=0,
            )

            # Decorate subplot
            self.set_xlabel("Model score", fontsize=12)
            self.set_ylabel("Counts", fontsize=12)
            self.set_title("Distribution of model score", fontsize=16)

            if log_y:
                self.set_yscale("log")

        # Store models and set plot type
        self._used_models = models
        self._plot_type = 'score_hist'

    def performance_curve(
        self,
        curve_type: str = "ROC",
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        log_x: Optional[bool] = False,
        get_background: Optional[bool] = False,
    ):
        # Get curve type and label configuration
        (
            _,
            _,
            x_label,
            y_label,
        ) = curve_config_dict[curve_type]

        # Get correct target label
        if not get_background:
            target = self._plotter._target_label
        else:
            target = self._plotter._background_label

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self._plotter.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        # Loop over each model and benchmark and add data to plot
        for m in models + benchmarks:
            if m is not None:
                model = m if not get_background else m.get_background_model()

                # Add data to plot
                x_rate, y_rate, _, auc = model.get_performance_curve(curve_type)
                self.plot(
                    x_rate,
                    y_rate,
                    color=model._color,
                    label=model._label + " - AUC = %.6s" % auc,
                    linestyle="solid",
                )

        # Add plot style and info
        self.set_axisbelow(True)
        self.grid(linestyle="dotted")
        self.set_xlabel(x_label, fontsize=12)
        self.set_ylabel(y_label, fontsize=12)
        self.set_title("Model performance on {}".format(target), fontsize=16)
        self.set_ylim(-0.02, 1.02)
        self.legend()
        if log_x:
            self.set_xscale("log")
        else:
            self.set_xlim(-0.02, 1.02)
        if curve_type == "ROC" and not log_x:
            self.plot([0, 1], [0, 1], color="k", linestyle="dashed", linewidth=1)

        # Set plot type
        self._plot_type = 'performance_curve'
   
    def score_by_energy(
        self,
        model_names: Optional[List[str]] = None,
        shift_y: Optional[bool] = False,
    ):
        # Add the correct models
        models = self._plotter.get_models_by_names(model_names)

        # Loop over models and add data to plot
        for model in models:
            if model._energy is None:
                print("{}: No energy found. Skipping.".format(model._name))
                continue

            m_ones = model._predictions[np.where(model._truths == 1)]
            m_zeros = model._predictions[np.where(model._truths == 0)]
            e_ones = model._energy[np.where(model._truths == 1)]
            e_zeros = model._energy[np.where(model._truths == 0)]

            # Get correct opacity for plotting
            alpha = calculate_alpha(model._predictions)

            self.scatter(
                e_ones,
                m_ones,
                color=model._color,
                marker=".",
                alpha=alpha,
                label=self._plotter._target_label + r" " + model._label,
            )
            self.scatter(
                e_zeros,
                m_zeros,
                color="k",
                marker=".",
                alpha=alpha,
                label=self._plotter._background_label + r" " + model._label,
            )

        # Force different y-axis if the data has outliers
        self.shift_axis(m_ones, m_zeros, shift_y=shift_y)

        # Decorate plot
        self.set_axisbelow(True)
        self.grid(linestyle="dotted")
        self.set_xlabel("Energy [MeV]", fontsize=12)
        self.set_ylabel("Model score", fontsize=12)
        self.set_title(
            "Distribution of model score by lepton energy", fontsize=16
        )
        leg = self.legend(
            fontsize=12,
            loc="upper center",
            bbox_to_anchor=[0.5, 0.90],
            markerscale=2,
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        # Store models and set plot type
        self._used_models = models
        self._plot_type, self._horizontal = 'score_by_energy', True

    def score_comparison(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        background: Optional[bool] = False,
        shift_x: Optional[bool] = True,
        shift_y: Optional[bool] = True,
    ):
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self._plotter.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        # Loop over models and add data to plot
        for model, benchmark in zip(models, benchmarks):
            if benchmark is None:
                print("{}: No benchmark model found. Skipping.".format(model._name))
                continue

            # Configure for background
            target, label = (
                (1, self._plotter._target_label) 
                if not background else 
                (0, self._plotter._background_label)
            )

            m_preds = model._predictions[np.where(model._truths == target)]
            b_preds = benchmark._predictions[np.where(benchmark._truths == target)]
            
            # Get correct opacity for plotting
            alpha = calculate_alpha(model._predictions)

            # Plot
            self.scatter(
                b_preds,
                m_preds,
                color=model._color,
                marker=".",
                alpha=alpha,
                label=label,
            )

        # Decorate plot
        self.set_axisbelow(True)
        self.grid(linestyle="dotted")
        self.set_xlabel(benchmark._label, fontsize=12)
        self.set_ylabel(model._label, fontsize=12)
        self.set_title("Distribution of model score", fontsize=16)
        leg = self.legend(fontsize=12, markerscale=2, loc="center right")
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        # Force shifted axes if the data has outliers
        self.shift_axis(b_preds, b_preds, shift_x=shift_x)
        self.shift_axis(m_preds, m_preds, shift_y=shift_y)

        # Store models and set plot type
        self._used_models = models + benchmarks
        self._plot_type = 'score_comparison'

    def score_by_position(
        self,
        dimensions: Tuple[int],
        model_names: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        get_background: Optional[bool] = False,
        get_correct: Optional[bool] = True,
        get_discarded: Optional[bool] = False,
    ):
        # Add the correct models
        models = self._plotter.get_models_by_names(model_names)

        if thresholds == None:
            thresholds = [None] * len(models)

        # Loop over models with the respective cuts
        for model, cuts in zip(models, thresholds):
            if model is None:
                continue
            if model._lepton_pos is None:
                print("{}: No position found. Skipping.".format(model._name))
                continue

            # Get the correct cuts for the target rates
            if cuts is None:
                model.calculate_target_rates()
                cuts = [
                    model._performance_rates[model._target_curve_type][0][2],
                    1
                    - model.get_background_model()._performance_rates[
                        model._target_curve_type
                    ][0][2],
                ]

            # Get true, false and discarded rates and label
            if not get_background:
                truth_bool = model._truths == 1

                if get_correct:
                    label = self._plotter._target_label
                    pred_bool = model._predictions > cuts[0]
                else:
                    label = self._plotter._target_label + " (" + self._plotter._background_label + r"$^{ID}$)"
                    pred_bool = model._predictions < cuts[1]

            else:
                truth_bool = model._truths == 0

                if get_correct:
                    label = self._plotter._background_label
                    pred_bool = model._predictions < cuts[1]
                else:
                    label = self._plotter._background_label + " (" + self._plotter._target_label + r"$^{ID}$)"
                    pred_bool = model._predictions > cuts[0]

            # Override if discarded is needed
            if get_discarded:
                pred_bool = (
                    model._predictions > cuts[1]) & (model._predictions < cuts[0]
                )
                label = 'Discarded'
                
            position = model._lepton_pos[np.where(truth_bool & pred_bool)]

            # Get correct opacity for plotting
            alpha = calculate_alpha(position)

            # Plot
            color = 'k'
            self.scatter(
                position[:, dimensions[0]],
                position[:, dimensions[1]],
                color=color,
                label=label,
                marker=".",
                alpha=alpha,
            )

        dim_labels = ['x', 'y', 'z']
        self.set_xlabel(dim_labels[dimensions[0]]+' [cm]', fontsize=12)
        self.set_ylabel(dim_labels[dimensions[1]]+' [cm]', fontsize=12)

        leg = self.legend(loc="upper right")
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        # Set plot type
        self._plot_type = 'score_by_position'

    def score_hist_by_distance(
        self,
        model_names: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        bins: Optional[int] = 20,
        range: Optional[List[float]] = None,
    ):
        # Add the correct models and benchmarks if not supplied
        models = self._plotter.get_models_by_names(model_names)

        if thresholds == None:
            thresholds = [None] * len(models)

        # Store data to use by ratio plots
        self._model_labels, self._model_counts, self._model_bins = [], [], []

        # Loop over models with the respective cuts
        for model, cuts in zip(models, thresholds):
            if model._lepton_pos is None:
                print("{}: No position found. Skipping.".format(model._name))
                continue

            # Get true and false rates
            if cuts is None:
                model.calculate_target_rates()
                cuts = [
                    model._performance_rates[model._target_curve_type][0][2],
                    1
                    - model.get_background_model()._performance_rates[
                        model._target_curve_type
                    ][0][2],
                ]

            pos_target_true = model._lepton_pos[
                np.where((model._truths == 1) & (model._predictions > cuts[0]))
            ]
            pos_target_false = model._lepton_pos[
                np.where((model._truths == 1) & (model._predictions < cuts[1]))
            ]
            pos_background_true = model._lepton_pos[
                np.where((model._truths == 0) & (model._predictions < cuts[1]))
            ]
            pos_background_false = model._lepton_pos[
                np.where((model._truths == 0) & (model._predictions > cuts[0]))
            ]

            positions_list = [
                [pos_target_true, pos_target_false],
                [pos_background_true, pos_background_false],
            ]
            labels_list = [
                [
                    self._plotter._target_label,
                    self._plotter._target_label
                    + " ("
                    + self._plotter._background_label
                    + r"$^{ID}$)",
                ],
                [
                    self._plotter._background_label,
                    self._plotter._background_label
                    + " ("
                    + self._plotter._target_label
                    + r"$^{ID}$)",
                ],
            ]

            # Plot
            for positions, labels, colors in zip(
                positions_list,
                labels_list,
                np.array(self._plotter._color_dict["particles"][:4]).reshape(2, -1),
            ):
                radii_true = np.sqrt(
                    positions[0][:, 0] ** 2 + positions[0][:, 1] ** 2
                )
                radii_false = np.sqrt(
                    positions[1][:, 0] ** 2 + positions[1][:, 1] ** 2
                )

                counts_true, bins_true = np.histogram(
                    radii_true, bins=bins, range=range
                )
                counts_false, _ = np.histogram(radii_false, bins=bins_true)
                bin_centers = bins_true[:-1] + (bins_true[1] - bins_true[0]) / 2

                self.stairs(
                    counts_true, bins_true, color=colors[0], label=labels[0]
                )
                self.stairs(
                    counts_false, bins_true, color=colors[1], label=labels[1]
                )

                self._model_counts.append([counts_true, counts_false])
                self._model_labels.append(labels[0])
                self._model_bins.append(bin_centers)

        # Decorate plot
        self.grid(linestyle="dotted")
        self.legend(fontsize=12)

        self.set_title("False prediction ratio - " + model._name, fontsize=16)

        self.set_ylim(0)
        self.grid(linestyle="dotted")
        self.set_xlabel("R [cm]", fontsize=12)
        self.set_ylabel("Counts", fontsize=12)
        self.legend(fontsize=12)

        # Set plot type
        self._plot_type = 'score_hist_by_distance'

    def score_ratio_by_distance(
        self,
        link_axis: Optional[RPPAxes] = None,
    ):
        if link_axis is None:
            link_axis = self.get_axis_neighbour(direction='above')

        for counts, label, bin_centers, color_comp in zip(
            link_axis._model_counts, 
            link_axis._model_labels,
            link_axis._model_bins,
            self._plotter._color_dict["compare"],
        ):
            counts_false, counts_true = counts[0], counts[1]

            ratios = counts_false / counts_true

            self.plot(
                bin_centers,
                ratios,
                color=color_comp,
                marker=".",
                ls="dotted",
                label=label,
            )

        self.legend(fontsize=12)
        self.set_xlabel("R [cm]", fontsize=12)
        self.set_ylabel("Ratio", fontsize=12)

        # Set plot type
        self._plot_type = 'score_ratio_by_distance'

    def add_cut_info(
        self,
        annotate: Optional[bool] = True,
        background: Optional[bool] = False,
    ):
        assert self._plot_type in ['score_hist', 'score_by_energy', 'score_comparison'], 'Only plot with model scores can have cuts and rates.'

        for i, model in enumerate(self._used_models):

            assert model._target_rates is not None or model._target_cuts is not None, 'One of the displayed models have no target parameters.'

            # Flip half the models if score comparison plot
            if self._plot_type == 'score_comparison' and i < len(self._used_models)/2:
                flip = True
            else:
                flip = False

            # Get rate data
            model.calculate_target_rates()
            threshold = model._performance_rates[model._target_curve_type][0][2]

            # Reverse threshold if background, get correct label and remove math mode
            label, threshold = (
                (self._plotter._background_label[1:-1], 1 - threshold)
                if background
                else (self._plotter._target_label[1:-1], threshold)
            )

            self.dynaline(
                threshold,
                flip=flip,
                c=self._plotter._color_dict["annotate"],
                zorder=3,
                **self._plotter._style_dict["annotate"],
            )

            # Add text to plot
            if annotate:
                text = []
                for function in model.get_performance_iterator(label, background):
                    if function._checkpoint:
                        text.append(function._name)
                        text.append(
                            r"TPR$_{%s} = %.2f$"
                            % (label, function._performance_rates[1] * 100)
                            + "%"
                        )
                        text.append(
                            r"FPR$_{%s} = %.2f$"
                            % (label, function._performance_rates[0] * 100)
                            + "%"
                        )

                textstr = "\n".join(tuple(text))
                props = dict(
                    boxstyle="square",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="lightgray",
                    pad=0.5,
                )
                self.text(
                    self._plotter._pos_dict[self._horizontal][background][0],
                    self._plotter._pos_dict[self._horizontal][background][1],
                    textstr,
                    transform=self.transAxes,
                    fontsize=12,
                    va="top",
                    bbox=props,
                    zorder=6,
                )