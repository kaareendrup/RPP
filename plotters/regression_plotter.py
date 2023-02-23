
import numpy as np
import matplotlib.pyplot as plt

from RPP.plotters.plotter import Plotter
from RPP.data.models import Model
from RPP.utils.maths.maths import bin_residual_width, w_errorprop
from RPP.utils.utils import make_colormap, basic_color_dict


class RegressionPlotter(Plotter):

    def __init__(self, plot_dir, target, unit, color_dict=basic_color_dict, cmap=make_colormap()):
        super().__init__(plot_dir, target, color_dict, cmap)
        
        self._unit = unit
        self._unit_str = " [" + self._unit +"]"

        self._bin_positions = []
        self._bin_centers = []
        self._n_residual_bins = []


    def plot_2Dhist(self, bins=100):
        for model_list in [self._models_list, self._benchmarks_list]:
            for model in model_list:

                # Plot 2D histogram and scatter plot of the same range
                fig, axs = plt.subplots(
                    1, 2, 
                    figsize=(18,7), 
                    sharey=True, 
                    gridspec_kw={'wspace': .04, 'width_ratios': [1,1.2]}
                )

                axs[0].scatter(model._predictions, model._truths, color=model._color, marker='.')
                h = axs[1].hist2d(model._predictions, model._truths, bins=bins, cmap=self._cmap)
                plt.colorbar(h[3], pad=0.01)

                axs[0].set_xlabel(self._target + r"$_\mathrm{true}$" + self._unit_str, size=12)
                axs[1].set_xlabel(self._target + r"$_\mathrm{true}$" + self._unit_str, size=12)
                axs[0].set_ylabel(self._target + r"$_\mathrm{reco}$" + self._unit_str, size=12)

                axs[0].set_title('Model performance on {}'.format(self._target), fontsize=12)
                axs[1].set_title('Model performance on {}'.format(self._target), fontsize=12)
                plt.savefig(self._plot_dir + model._name + "_hist2D.png")
                plt.close()


    def get_residual_widths(self, model, verbose=False, relative=False):

        avg_res_list, w_list, w_err_list, not_conv = [], [], [], []

        # Calculate resdiual distribution width w for each bin and check convergence
        for i in np.arange(self._n_residual_bins)+1:

            # Get all data points in the bin
            bin_truths = model._truths[np.where(self._bin_positions==i)]
            bin_preds = model._predictions[np.where(self._bin_positions==i)]

            if relative:
                bin_truths, bin_preds = bin_truths/bin_truths, bin_preds/bin_truths
            
            # Calculate bin residual width, error and average bin residual
            bin_avg_res, bin_w, w_err, converged = bin_residual_width(bin_truths, bin_preds, verbose)
            w_list.append(bin_w)
            w_err_list.append(w_err)
            avg_res_list.append(bin_avg_res)
            if (not converged):
                not_conv.append(i-1)

        model._avg_res, model._w, model._w_err = avg_res_list, np.array(w_list), np.array(w_err_list)

        # Export non converged points for troubleshooting
        if len(not_conv) > 0:
            model._not_conv_x = np.array(self._bin_centers)[not_conv]
            model._not_conv_y = np.array(w_list)[not_conv]
        else: 
            model._not_conv_x, model._not_conv_y = np.array([]), np.array([])


    def calc_diff(self, model, benchmark, color):

        # Calculate diffs and propagate errors
        diffs_w = (benchmark._w-model._w)/benchmark._w
        diffs_err = w_errorprop(benchmark._w, model._w, benchmark._w_err, model._w_err)

        # Create placeholder model and add values
        compare_model = Model(model._name, None, None, model._event_nos, color)
        compare_model._w, compare_model._w_err = diffs_w, diffs_err
        compare_model._not_conv_x, compare_model._not_conv_y = np.array([]), np.array([])

        return compare_model


    def add_data_to_plot(self, axis, model):
        
        # Plot residual widths and unconverged values
        axis.errorbar(
            self._bin_centers, 
            model._w, 
            yerr=model._w_err, 
            c=model._color, 
            marker='.',
            linestyle='dotted', 
            markersize=8, 
            capsize=8, 
            label=model._name
        )
        axis.scatter(model._not_conv_x, model._not_conv_y, c='r', s=60)


    def plot_resolution(self, model_names=None, benchmark_names=None, n_residual_bins=10):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Initialize subplots and invert so it can be looped
        n_axs, h_ratios = (1, [1]) if benchmarks[0] == None else (2, [5,1])
        _, subplots = plt.subplots(
            n_axs, 2,
            sharex=True,
            figsize=(18,7),
            gridspec_kw={'wspace': 0.3, 'hspace': 0.05, 'height_ratios': h_ratios}
        )
        hist_ax = np.array([ax.twinx() for ax in subplots[0]])[np.newaxis,:]
        subplots = np.append(subplots, hist_ax, axis=0).transpose()

        # Make the bins to calculate w for
        self._n_residual_bins = n_residual_bins
        _, self._bin_edges = np.histogram(models[0]._truths, bins=self._n_residual_bins)
        self._bin_positions = np.digitize(models[0]._truths, self._bin_edges, right=True)
        self._bin_centers = self._bin_edges[:-1] + (self._bin_edges[1]-self._bin_edges[0])/2

        # Loop over each model and relative/absolute plots
        for subplot, relative, label in zip(subplots, [False, True], ['absolute' + self._unit_str, 'relative']):
            
            # Use shorthand
            ax_top, ax_back, ax_bot = subplot[0], subplot[-1], subplot[-2]

            for model, benchmark, compare_color in zip(models, benchmarks, self._color_dict['compare']):

                # Add data
                self.get_residual_widths(model, relative=relative)
                self.add_data_to_plot(ax_top, model)

                # Add benchmark
                if benchmark is not None:
                    self.get_residual_widths(benchmark, relative=relative)
                    self.add_data_to_plot(ax_top, benchmark)

                    # Add compare plot
                    compare_model = self.calc_diff(model, benchmark, compare_color)
                    self.add_data_to_plot(ax_bot, compare_model)
                    ax_bot.axhline(0, color='dimgray', linewidth=1)

            # Add plot style and info
            ax_bot.set_xlabel(self._target + r"$_\mathrm{true}$" + self._unit_str, fontsize=12)
            ax_top.set_ylabel(r'$\sigma _\mathrm{' + label + '}$', fontsize=12)
            ax_back.hist(model._truths, bins=n_residual_bins, color='lightgray')
            ax_back.set_ylabel('Counts', fontsize=12)
            ax_top.legend()

            ax_top.set_title('Model performance on {}'.format(self._target), fontsize=12)
            ax_top.set_zorder(ax_back.get_zorder()+1)
            ax_top.patch.set_visible(False)

        plt.savefig(self._plot_dir + model._name + "_resolution.png")
        plt.close()

