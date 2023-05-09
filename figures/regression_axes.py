from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from RPP.figures.axes import RPPAxes
from RPP.data.models import Model
from RPP.utils.maths.maths import bin_residual_width, w_errorprop

class RegressionAxes(RPPAxes):

    def hist2D(
        self, 
        model: Model,
        bins: Optional[int] = 100
    ):

        h = self.hist2d(
            model._predictions, model._truths, bins=bins, cmap=self._plotter._cmap
        )
        plt.colorbar(h[3], pad=0.01)

        self.set_xlabel(
            self._plotter._target + r"$_\mathrm{true}$" + self._plotter._unit_str, size=12
        )
        self.set_ylabel(
            self._plotter._target + r"$_\mathrm{reco}$" + self._plotter._unit_str, size=12
        )

        self.set_title(
            "Model performance on {}".format(self._plotter._target), fontsize=12
        )

    def scatter2D(
        self, 
        model: Model,
    ):

        self.scatter(
            model._predictions, model._truths, color=model._color, marker="."
        )

        self.set_xlabel(
            self._plotter._target + r"$_\mathrm{true}$" + self._plotter._unit_str, size=12
        )
        self.set_ylabel(
            self._plotter._target + r"$_\mathrm{reco}$" + self._plotter._unit_str, size=12
        )

        self.set_title(
            "Model performance on {}".format(self._plotter._target), fontsize=12
        )

    def get_residual_widths(
        self,
        model: Model,
        verbose: Optional[bool] = False,
        relative: Optional[bool] = False,
    ):
        avg_res_list, w_list, w_err_list, not_conv = [], [], [], []

        # Calculate resdiual distribution width w for each bin and check convergence
        for i in np.arange(self._n_residual_bins) + 1:
            # Get all data points in the bin
            bin_truths = model._truths[np.where(self._bin_positions == i)]
            bin_preds = model._predictions[np.where(self._bin_positions == i)]

            if relative:
                bin_truths, bin_preds = bin_truths / bin_truths, bin_preds / bin_truths

            # Calculate bin residual width, error and average bin residual
            bin_avg_res, bin_w, w_err, converged = bin_residual_width(
                bin_truths, bin_preds, verbose
            )
            w_list.append(bin_w)
            w_err_list.append(w_err)
            avg_res_list.append(bin_avg_res)
            if not converged:
                not_conv.append(i - 1)

        model._avg_res, model._w, model._w_err = (
            avg_res_list,
            np.array(w_list),
            np.array(w_err_list),
        )

        # Export non converged points for troubleshooting
        if len(not_conv) > 0:
            model._not_conv_x = np.array(self._bin_centers)[not_conv]
            model._not_conv_y = np.array(w_list)[not_conv]
        else:
            model._not_conv_x, model._not_conv_y = np.array([]), np.array([])

    def resolution(
        self,
        models: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        n_residual_bins: Optional[int] = 10,
        relative: Optional[bool] = False,
    ):
        # Create histogram axis
        self._ax_bg = self.twinx()

        label = "relative" if relative else "absolute" + self._plotter._unit_str

        # Add the correct models and benchmarks if not supplied
        self._models, self._benchmarks = self._plotter.get_models_and_benchmarks(
            models, benchmarks
        )
        
        # Make the bins to calculate w for
        self._n_residual_bins = n_residual_bins
        _, self._bin_edges = np.histogram(self._models[0]._truths, bins=self._n_residual_bins)
        self._bin_positions = np.digitize(
            self._models[0]._truths, self._bin_edges, right=True
        )
        self._bin_centers = (
            self._bin_edges[:-1] + (self._bin_edges[1] - self._bin_edges[0]) / 2
        )

        for model, benchmark in zip(self._models, self._benchmarks):
            # Add data
            self.get_residual_widths(model, relative=relative)
            self.errorbar(
                self._bin_centers,
                model._w,
                yerr=model._w_err,
                c=model._color,
                marker=".", # Gather these in **kwargs
                linestyle="dotted",
                markersize=8,
                capsize=8,
                label=model._name,
            )
            self.scatter(model._not_conv_x, model._not_conv_y, c="r", s=60)

            # Add benchmark
            if benchmark is not None:
                self.get_residual_widths(benchmark, relative=relative)
                self.errorbar(
                    self._bin_centers,
                    benchmark._w,
                    yerr=benchmark._w_err,
                    c=benchmark._color,
                    marker=".",
                    linestyle="dotted",
                    markersize=8,
                    capsize=8,
                    label=benchmark._name,
                )
                self.scatter(benchmark._not_conv_x, benchmark._not_conv_y, c="r", s=60)

        # Decorate plot
        self.set_xlabel(
            self._plotter._target + r"$_\mathrm{true}$" + self._plotter._unit_str, fontsize=12
        )
        self.set_ylabel(r"$\sigma _\mathrm{" + label + "}$", fontsize=12)
        self._ax_bg.hist(model._truths, bins=n_residual_bins, color="lightgray")
        self._ax_bg.set_ylabel("Counts", fontsize=12)
        self.legend()

        self.set_title(
            "Model performance on {}".format(self._plotter._target), fontsize=12
        )
        self.set_zorder(self._ax_bg.get_zorder() + 1)
        self.patch.set_visible(False)

        plt.savefig(self._plotter._plot_dir + model._name + "_resolution.png")
        plt.close()

    def resolution_ratio(
        self,
        link_axis: Optional[RPPAxes] = None,
    ):
        if link_axis is None:
            link_axis = self.get_axis_neighbour(direction='above')

        for model, benchmark, compare_color in zip(
            link_axis._models, link_axis._benchmarks, self._plotter._color_dict["compare"]
        ):
            
            # Calculate diffs and propagate errors
            diffs_w = (benchmark._w - model._w) / benchmark._w
            diffs_err = w_errorprop(benchmark._w, model._w, benchmark._w_err, model._w_err)

            self.errorbar(
                link_axis._bin_centers,
                diffs_w,
                yerr=diffs_err,
                c=compare_color,
                marker=".",
                linestyle="dotted",
                markersize=8,
                capsize=8,
                label=model._name,
            )

        self.axhline(0, color="dimgray", linewidth=1)
        
        # Decorate plot
        self.set_xlabel(
            self._plotter._target + r"$_\mathrm{true}$" + self._plotter._unit_str, fontsize=12
        )
