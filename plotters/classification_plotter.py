from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from RPP.plotters.plotter import Plotter
from RPP.data.models import ClassificationModel, Model
from RPP.utils.utils import (
    calculate_alpha,
    beautify_label,
    add_rates,
    curve_config_dict,
    shift_axis,
)
from RPP.utils.style import basic_pos_dict
from RPP.utils.data import Cutter


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
        self._background = background
        self._background_label = beautify_label(background)
        self._show_cuts = show_cuts
        self._pos_dict = pos_dict

    def add_rate_info(
        self,
        axs: List[Axes],
        model: ClassificationModel,
        horizontal: Optional[bool] = False,
        annotate: Optional[bool] = True,
        plot_sig: Optional[bool] = True,
        plot_bg: Optional[bool] = True,
    ):
        for m, is_bg, plot in zip(
            [model, model.get_background_model()], [False, True], [plot_sig, plot_bg]
        ):
            if (
                m._target_rates is not None or m._target_cuts is not None
            ) and plot:
                # Get rate data
                m.calculate_target_rates()
                threshold = m._performance_rates[m._target_curve_type][0][2]

                # Reverse threshold if background, get correct label and remove math mode
                label, threshold = (
                    (self._background_label[1:-1], 1 - threshold)
                    if is_bg
                    else (self._target_label[1:-1], threshold)
                )

                # Add cut line for the correct curve type
                for ax in axs:
                    if not horizontal:
                        ax.axvline(
                            threshold,
                            c=self._color_dict["annotate"],
                            zorder=3,
                            **self._style_dict["annotate"]
                        )
                    else:
                        ax.axhline(
                            threshold,
                            c=self._color_dict["annotate"],
                            zorder=3,
                            **self._style_dict["annotate"]
                        )

                # Add text to plot
                if annotate:
                    text = []
                    for function in m.get_performance_iterator(label, is_bg):
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
                    axs[0].text(
                        self._pos_dict[horizontal][is_bg][0],
                        self._pos_dict[horizontal][is_bg][1],
                        textstr,
                        transform=axs[0].transAxes,
                        fontsize=12,
                        va="top",
                        bbox=props,
                        zorder=6,
                    )

    def plot_score_hist(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        n_bins: Optional[int] = 100,
        shift_x: Optional[bool] = False,
        show_rate_info: Optional[bool] = True, 
    ):
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        for model_list in [models, benchmarks]:
            if model_list[0] is not None:
                # Plot 2D histogram and scatter plot of the same range
                _, axs = plt.subplots(
                    1,
                    2,
                    figsize=(18, 7),
                    gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1]},
                )

                for model, ls in zip(model_list, self._style_dict["histogram"]):
                    # Subdivide in truth and background
                    ones = model._predictions[np.where(model._truths == 1)]
                    zeros = model._predictions[np.where(model._truths == 0)]

                    # Plot a regular and logged histogram
                    for log, ax in zip([False, True], axs):
                        if shift_x:
                            ax, xmin, xmax = shift_axis(ax, ones, zeros, shift_x=True)
                            bins = np.linspace(xmin, xmax, n_bins)
                        else:
                            bins = n_bins

                        ax.hist(
                            ones,
                            color=model._color,
                            bins=bins,
                            histtype="step",
                            linestyle=ls,
                            label=self._target_label + r" " + model._label,
                            zorder=1,
                        )
                        ax.hist(
                            zeros,
                            color="k",
                            bins=bins,
                            histtype="step",
                            linestyle=ls,
                            label=self._background_label + r" " + model._label,
                            zorder=0,
                        )

                        # Decorate subplot
                        ax.set_xlabel("Model score", fontsize=12)
                        ax.set_ylabel("Counts", fontsize=12)
                        ax.set_title("Distribution of model score", fontsize=16)

                        if log:
                            ax.set_yscale("log")

                    self.add_rate_info(axs, model, annotate=show_rate_info)

                # Decorate plot
                leg_loc = (
                    (0.085, 0.85)
                    if len(models) == 1
                    else (0.085, 0.85 - (len(models) - 1) * 0.1)
                )
                axs[0].legend(fontsize=12, loc=leg_loc)

                plt.savefig(
                    self._plot_dir
                    + models[0]._title
                    + "_clfhist_"
                    + model_list[0]._name
                    + ".png"
                )
                plt.close()

    def plot_performance_curve(
        self,
        curve_type: str = "ROC",
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        log_x: Optional[bool] = False,
        get_background: Optional[bool] = False,
    ):
        # Get curve type and label configuration
        curve_config = curve_config_dict[curve_type]
        target, title = (
            (self._target_label, curve_type)
            if not get_background
            else (self._background_label, curve_type + "_BG")
        )

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        # Initialize subplots and invert so it can be looped
        n_axs, size, kw = (
            (1, (9, 7), None) if not log_x else (2, (18, 7), {"wspace": 0.3})
        )
        _, (axs,) = plt.subplots(
            1,
            n_axs,
            figsize=size,
            squeeze=False,
            gridspec_kw=kw,
        )

        # Loop over each model and benchmark and add data to plot
        for m in models + benchmarks:
            if m is not None:
                model = m if not get_background else m.get_background_model()

                # Add data to plot
                for ax in axs:
                    x_rate, y_rate, _, auc = model.get_performance_curve(curve_type)
                    ax.plot(
                        x_rate,
                        y_rate,
                        color=model._color,
                        label=model._label + " - AUC = %.6s" % auc,
                        linestyle="solid",
                    )

                if model._target_rates is not None:
                    model.calculate_target_rates(curve_type)
                    add_rates(axs, model, curve_type)

        # Add plot style and info
        for ax in axs:
            ax.set_axisbelow(True)
            ax.grid(linestyle="dotted")
            ax.set_xlabel(curve_config['x_label'], fontsize=12)
            ax.set_title("Model performance on {}".format(target), fontsize=12)
            ax.set_ylim(-0.02, 1.02)

        axs[0].set_ylabel(curve_config['y_label'], fontsize=12)
        axs[0].set_xlim(-0.02, 1.02)
        axs[-1].legend()
        if curve_type == "ROC":
            axs[0].plot([0, 1], [0, 1], color="k", linestyle="dashed", linewidth=1)
        if log_x:
            axs[1].set_xscale("log")

        plt.savefig(self._plot_dir + models[0]._title + "_" + title + ".png")
        plt.close()

    def plot_score_by_energy(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        shift_y: Optional[bool] = False,
        show_rate_info: Optional[bool] = True,
    ):
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        for model, benchmark in zip(models, benchmarks):
            _, (axs,) = plt.subplots(
                1,
                1 + (benchmark is not None),
                figsize=(9 * (1 + (benchmark is not None)), 7),
                gridspec_kw={"wspace": 0.3},
                squeeze=False,
            )

            # Loop over models and add data to plot
            for m, ax in zip([model, benchmark], axs):
                if m._energy is None:
                    print("{}: No energy found. Skipping.".format(m._name))
                    continue

                m_ones = m._predictions[np.where(m._truths == 1)]
                m_zeros = m._predictions[np.where(m._truths == 0)]
                e_ones = m._energy[np.where(m._truths == 1)]
                e_zeros = m._energy[np.where(m._truths == 0)]

                # Get correct opacity for plotting
                alpha = calculate_alpha(m._predictions)

                ax.scatter(
                    e_ones,
                    m_ones,
                    color=m._color,
                    marker=".",
                    alpha=alpha,
                    label=self._target_label + r" " + m._label,
                )
                ax.scatter(
                    e_zeros,
                    m_zeros,
                    color="k",
                    marker=".",
                    alpha=alpha,
                    label=self._background_label + r" " + m._label,
                )

                # Force different y-axis if the data has outliers
                if shift_y:
                    ax, _, _ = shift_axis(ax, m_ones, m_zeros, shift_y=True)

                # Add rate info to plot
                self.add_rate_info([ax], m, horizontal=True, annotate=show_rate_info)

                # Decorate plot
                ax.set_axisbelow(True)
                ax.grid(linestyle="dotted")
                ax.set_xlabel("Energy [MeV]", fontsize=12)
                ax.set_ylabel("Model score", fontsize=12)
                ax.set_title(
                    "Distribution of model score by lepton energy", fontsize=16
                )
                leg = ax.legend(
                    fontsize=12,
                    loc="upper center",
                    bbox_to_anchor=[0.5, 0.90],
                    markerscale=2,
                )
                for lh in leg.legendHandles:
                    lh.set_alpha(1)

            plt.savefig(self._plot_dir + model._title + "_energy_score.png")
            plt.close()

    def plot_score_comparison(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        shift_axes: Optional[bool] = True,
    ):
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        # Loop over models and add data to plot
        for model, benchmark in zip(models, benchmarks):
            if benchmark is None:
                print("{}: No benchmark model found. Skipping.".format(model._name))
                continue

            _, axs = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"wspace": 0.3})

            m_ones = model._predictions[np.where(model._truths == 1)]
            m_zeros = model._predictions[np.where(model._truths == 0)]
            b_ones = benchmark._predictions[np.where(benchmark._truths == 1)]
            b_zeros = benchmark._predictions[np.where(benchmark._truths == 0)]
            predictions = [[b_ones, m_ones], [b_zeros, m_zeros]]

            # Get correct opacity for plotting
            alpha = calculate_alpha(model._predictions)

            for ax, preds, label, color, is_sig in zip(
                axs,
                predictions,
                [self._target_label, self._background_label],
                [model._color, "k"],
                [True, False],
            ):
                ax.scatter(
                    preds[0],
                    preds[1],
                    color=color,
                    marker=".",
                    alpha=alpha,
                    label=label,
                )

                # Decorate plot
                ax.set_axisbelow(True)
                ax.grid(linestyle="dotted")
                ax.set_xlabel(benchmark._label, fontsize=12)
                ax.set_title("Distribution of model score", fontsize=16)
                leg = ax.legend(fontsize=12, markerscale=2, loc="center right")
                for lh in leg.legendHandles:
                    lh.set_alpha(1)

                # Add rates
                self.add_rate_info(
                    [ax], benchmark, annotate=False, plot_sig=is_sig, plot_bg=not is_sig
                )
                self.add_rate_info(
                    [ax],
                    model,
                    horizontal=True,
                    annotate=False,
                    plot_sig=is_sig,
                    plot_bg=not is_sig,
                )

            axs[0].set_ylabel(model._label, fontsize=12)

            # Force shifted axes if the data has outliers
            for ax in axs:
                if shift_axes:
                    ax, _, _ = shift_axis(ax, b_ones, b_zeros, shift_x=True)
                    ax, _, _ = shift_axis(ax, m_ones, m_zeros, shift_y=True)

            plt.savefig(self._plot_dir + model._title + "_model_scores.png")
            plt.close()

    def plot_score_by_position(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
    ):
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        if thresholds == None:
            thresholds = [None] * (len(models) + len(benchmarks))

        # Loop over models with the respective cuts
        for model, cuts in zip(models + benchmarks, thresholds):
            if model is None:
                continue
            if model._lepton_pos is None:
                print("{}: No position found. Skipping.".format(model._name))
                continue

            fig, axses = plt.subplots(
                2, 5, figsize=(24, 9), gridspec_kw={"wspace": 0.3, "hspace": 0.3}
            )

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

            # Get true, false and discarded rates
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
            pos_all_discarded = model._lepton_pos[
                np.where(
                    (model._predictions > cuts[1]) & (model._predictions < cuts[0])
                )
            ]

            positions = [
                pos_target_true,
                pos_target_false,
                pos_background_true,
                pos_background_false,
                pos_all_discarded,
            ]
            labels = [
                self._target_label,
                self._target_label + " (" + self._background_label + r"$^{ID}$)",
                self._background_label,
                self._background_label + " (" + self._target_label + r"$^{ID}$)",
                "Discarded",
            ]

            # Plot
            for axs, position, label, color in zip(
                axses.T, positions, labels, self._color_dict["particles"]
            ):
                # Get correct opacity for plotting
                alpha = calculate_alpha(position)

                axs[0].scatter(
                    position[:, 0],
                    position[:, 1],
                    color=color,
                    label=label,
                    marker=".",
                    alpha=alpha,
                )
                axs[1].scatter(
                    position[:, 0],
                    position[:, 2],
                    color=color,
                    label=label,
                    marker=".",
                    alpha=alpha,
                )

                axs[1].set_xlabel("x [cm]", fontsize=12)

                leg = axs[0].legend(loc="upper right")
                for lh in leg.legendHandles:
                    lh.set_alpha(1)

            axses.T[0, 0].set_ylabel("y [cm]", fontsize=12)
            axses.T[0, 1].set_ylabel("z [cm]", fontsize=12)

            fig.suptitle("Event position distributions", fontsize=16)

            plt.savefig(
                self._plot_dir
                + models[0]._title
                + "_scores_by_position_"
                + model._name
                + ".png"
            )
            plt.close()

    def plot_score_ratio_by_distance(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        bins: Optional[int] = 20,
        range: Optional[List[float]] = None,
    ):
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(
            model_names, benchmark_names
        )

        if thresholds == None:
            thresholds = [[None] * 2] * len(models)
        else:
            thresholds = np.array(thresholds).reshape(len(models), 2, 2)

        # Loop over models with the respective cuts
        for model, benchmark, cuts_list in zip(models, benchmarks, thresholds):
            if model._lepton_pos is None:
                print("{}: No position found. Skipping.".format(model._name))
                continue

            _, axses = plt.subplots(
                2,
                1 + (benchmark is not None),
                sharex=True,
                figsize=(9 * (1 + (benchmark is not None)), 8),
                gridspec_kw={"wspace": 0.3, "hspace": 0.05, "height_ratios": [3, 1]},
                squeeze=False,
            )

            for m, axs, cuts in zip([model, benchmark], axses.T, cuts_list):
                # Get true and false rates
                if cuts is None:
                    model.calculate_target_rates()
                    cuts = [
                        m._performance_rates[m._target_curve_type][0][2],
                        1
                        - m.get_background_model()._performance_rates[
                            m._target_curve_type
                        ][0][2],
                    ]

                pos_target_true = m._lepton_pos[
                    np.where((m._truths == 1) & (m._predictions > cuts[0]))
                ]
                pos_target_false = m._lepton_pos[
                    np.where((m._truths == 1) & (m._predictions < cuts[1]))
                ]
                pos_background_true = m._lepton_pos[
                    np.where((m._truths == 0) & (m._predictions < cuts[1]))
                ]
                pos_background_false = m._lepton_pos[
                    np.where((m._truths == 0) & (m._predictions > cuts[0]))
                ]

                positions_list = [
                    [pos_target_true, pos_target_false],
                    [pos_background_true, pos_background_false],
                ]
                labels_list = [
                    [
                        self._target_label,
                        self._target_label
                        + " ("
                        + self._background_label
                        + r"$^{ID}$)",
                    ],
                    [
                        self._background_label,
                        self._background_label
                        + " ("
                        + self._target_label
                        + r"$^{ID}$)",
                    ],
                ]

                # Plot
                for positions, labels, colors, color_comp in zip(
                    positions_list,
                    labels_list,
                    np.array(self._color_dict["particles"][:4]).reshape(2, -1),
                    self._color_dict["compare"],
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

                    axs[0].stairs(
                        counts_true, bins_true, color=colors[0], label=labels[0]
                    )
                    axs[0].stairs(
                        counts_false, bins_true, color=colors[1], label=labels[1]
                    )

                    ratios = counts_false / counts_true
                    bin_centers = bins_true[:-1] + (bins_true[1] - bins_true[0]) / 2

                    axs[1].plot(
                        bin_centers,
                        ratios,
                        color=color_comp,
                        marker=".",
                        ls="dotted",
                        label=labels[0],
                    )

                # Decorate plot
                axs[0].grid(linestyle="dotted")
                axs[0].legend(fontsize=12)

                axs[0].set_title("False prediction ratio - " + m._name, fontsize=16)

                axs[1].set_ylim(0)
                axs[1].grid(linestyle="dotted")
                axs[1].set_xlabel("R [cm]", fontsize=12)
                axs[1].legend(fontsize=12)

            axses[0, 0].set_ylabel("Counts", fontsize=12)
            axses[1, 0].set_ylabel("Ratio", fontsize=12)

            plt.savefig(self._plot_dir + model._title + "_scores_by_distance.png")
            plt.close()
