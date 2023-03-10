
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from RPP.plotters.plotter import Plotter
from RPP.utils.utils import make_colormap, basic_color_dict, basic_style_dict, curve_config_dict, calculate_alpha, beautify_label


class ClassificationPlotter(Plotter):

    def __init__(self, plot_dir, target, background, color_dict=basic_color_dict, style_dict=basic_style_dict, cmap=make_colormap(), cut_functions=None, show_cuts=True):
        super().__init__(plot_dir, target, color_dict, style_dict, cmap, cut_functions)

        self._background = background
        self._background_label = beautify_label(background)
        self._show_cuts = show_cuts


    def apply_rates_to_plot(self, axs, cut, xs, ys, is_bg, horizontal, annotate):

        # Get position
        pos_dict = {False: {False: [0.66, 0.82], True: [0.1, 0.82]}, True: {False: [0.73, 0.87], True: [0.73, 0.37]}}
        pos = pos_dict[horizontal][is_bg]

        # Reverse cut if background and remove math mode
        cut, label = (1-cut, self._background_label) if is_bg else (cut, self._target_label)
        label = label[1:-1]

        # Add cut line
        for ax in axs:
            if not horizontal:
                ax.axvline(cut, c=self._color_dict['annotate'], zorder=3, **self._style_dict['annotate'])
            else:
                ax.axhline(cut, c=self._color_dict['annotate'], zorder=3, **self._style_dict['annotate'])

        if annotate:
            # Add text to plot
            text = []
            for x, y, title in zip(xs, ys, [r'Cut$_{%s} = %.3f$' % (label, cut), (self._cut_functions[-1]._name)]):
                text.append(title)
                text.append(r'TPR$_{%s} = %.2f$' % (label, y*100) + '%')
                text.append(r'FPR$_{%s} = %.2f$' % (label, x*100) + '%')

            textstr = '\n'.join(tuple(text))
            props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray', pad=0.5)
            axs[0].text(pos[0], pos[1], textstr, transform=axs[0].transAxes, fontsize=12, va='top', bbox=props)


    def add_rate_info(self, axs, model, horizontal=False, annotate=True, plot_sig=True, plot_bg=True):
        
        # Loop over signal/background
        for model, is_bg, plot in zip([model, model.get_background_model()], [False, True], [plot_sig, plot_bg]):
            if plot:
                if model._target_rates is not None:

                    # Get threshold, fpr and tpr closest to target rate
                    rate = model._target_rates[-1]

                    fpr, tpr, thresholds = roc_curve(model._truths, model._predictions)
                    diffs = abs(fpr - (1-rate))
                    idx = np.argmin(diffs)
                    ifpr, itpr, ithreshold = fpr[idx], tpr[idx], thresholds[idx]

                    # Add rates adjusted for data cut TODO: I think this might be a little bit wrong, since the distribution pre/post cut are different
                    ratio = len(model._truths)/len(model._original_truths)
                    if ratio < 1 and self._show_cuts:
                        fprs, tprs = [ifpr, ifpr*ratio], [itpr, itpr*ratio]
                    else:
                        fprs, tprs = [ifpr], [itpr]

                    self.apply_rates_to_plot(axs, ithreshold, fprs, tprs, is_bg, horizontal, annotate)

                if model._target_cuts is not None:
                    cut = model._target_cuts[-1]
                    fprs, tprs = [], []

                    # Loop over with/without data cut if applicable
                    for truths, no_cut in [(model._truths, True), (model._original_truths, False)]: # TODO: Maybe cut this loop?
                        if no_cut or (len(truths) > len(model._truths) and self._show_cuts):

                            # Get signal and background before and after all cuts
                            ones = model._predictions[np.where(model._truths == 1)]
                            zeros = model._predictions[np.where(model._truths == 0)]
                            tprs.append(np.count_nonzero(ones>cut)/np.count_nonzero(truths == 1))
                            fprs.append(np.count_nonzero(zeros>cut)/np.count_nonzero(truths == 0))

                    self.apply_rates_to_plot(axs, cut, fprs, tprs, is_bg, horizontal, annotate)


    def plot_score_hist(self, model_names=None, benchmark_names=None, n_bins=100, shift_x=False):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        for model_list in [models, benchmarks]:
            for model in model_list:

                # Plot 2D histogram and scatter plot of the same range
                _, axs = plt.subplots(
                    1, 2, 
                    figsize=(18,7), 
                    gridspec_kw={'wspace': .3, 'width_ratios': [1,1]}
                )

                # Subdivide in truth and background
                ones = model._predictions[np.where(model._truths == 1)]
                zeros = model._predictions[np.where(model._truths == 0)]
                
                # Plot a regular and logged histogram
                for log, ax in zip([False, True], axs):

                    if shift_x:
                        xmax = max([np.percentile(ones, 99.9), np.percentile(zeros, 99.9)])
                        xmin = min([np.percentile(ones, .1), np.percentile(zeros, .1)])
                        xmean = (np.mean(ones)+np.mean(zeros))/2
                        dist = max([xmax-xmean, xmean-xmin])
                        ax.set_xlim(xmean-dist*1.1, xmean+dist*1.1)
                        bins = np.linspace(xmean-dist*1.1, xmean+dist*1.1, n_bins)
                    else:
                        bins = n_bins

                    ax.hist(ones, color=model._color, bins=bins, histtype='step', label=self._target_label +r' '+ model._label, zorder=1)
                    ax.hist(zeros, color='k', bins=bins, histtype='step', label=self._background_label +r' '+ model._label, zorder=0)

                    # Decorate subplot
                    ax.set_xlabel('Model score', fontsize=12)
                    ax.set_ylabel('Counts', fontsize=12)
                    ax.set_title('Distribution of model score', fontsize=16)

                    if log:
                        ax.set_yscale('log')

                # Decorate plot
                axs[0].legend(fontsize=12, loc=(0.085, 0.85))
                self.add_rate_info(axs, model)

                plt.savefig(self._plot_dir + model._title + "_clfhist.png")
                plt.close()


    def plot_performance_curve(self, curve_type='ROC', model_names=None, benchmark_names=None, log_x=False, get_background=False):

        # Get curve type and label configuration
        metric_function, metric_score, x_label, y_label, add_rates = curve_config_dict[curve_type]
        target, title = (self._target_label, curve_type) if not get_background else (self._background_label, curve_type+'_BG')

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Initialize subplots and invert so it can be looped
        n_axs, size, kw = (1, (9,7), None) if not log_x else (2, (18,7), {'wspace': 0.3})
        _, (axs,) = plt.subplots(
            1, n_axs,
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

                    # Get ROC/PR curve and auc score and plot
                    x_rate, y_rate, thresholds = metric_function(model._truths, model._predictions)
                    auc = metric_score(model._truths, model._predictions)
                    ax.plot(x_rate, y_rate, color=model._color, label=model._label + ' - AUC = %.6s'%auc, linestyle='solid')

                add_rates(axs, x_rate, y_rate, thresholds, model)

        # Add plot style and info
        for ax in axs:
            ax.set_axisbelow(True)
            ax.grid(linestyle='dotted')
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_title('Model performance on {}'.format(target), fontsize=12)
            ax.set_ylim(-.02,1.02)

        axs[0].set_ylabel(y_label, fontsize=12)
        axs[0].set_xlim(-.02,1.02)
        axs[-1].legend()
        if curve_type == 'ROC':
            axs[0].plot([0,1], [0,1], color='k', linestyle='dashed', linewidth=1)
        if log_x:
            axs[1].set_xscale('log')

        plt.savefig(self._plot_dir + models[0]._title + "_" + title + ".png")
        plt.close()


    def plot_score_by_energy(self, model_names=None, benchmark_names=None, shift_y=False):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        for model, benchmark in zip(models, benchmarks):

            _, axs = plt.subplots(
                    1, 2, 
                    figsize=(18,7), 
                    gridspec_kw={'wspace': .3, 'width_ratios': [1,1]}
                )

            # Loop over models and add data to plot
            for m, ax in zip([model, benchmark], axs):
                if m._energy is None:
                    print("{}: No energy found. Skipping.".format(m._name))
                    continue

                else:
                    m_ones = m._predictions[np.where(m._truths == 1)]
                    m_zeros = m._predictions[np.where(m._truths == 0)]
                    e_ones = m._energy[np.where(m._truths == 1)]
                    e_zeros = m._energy[np.where(m._truths == 0)]

                    # Get correct opacity for plotting
                    alpha = calculate_alpha(m._predictions)

                    ax.scatter(e_ones, m_ones, color=m._color, marker='.', alpha=alpha, label=self._target_label +r' '+ m._label)
                    ax.scatter(e_zeros, m_zeros, color='k', marker='.', alpha=alpha, label=self._background_label +r' '+ m._label)

                    # Force different y-axis if the data has outliers
                    if shift_y:
                        ymax = max([np.percentile(m_ones, 99.9), np.percentile(m_zeros, 99.9)])
                        ymin = min([np.percentile(m_ones, .1), np.percentile(m_zeros, .1)])
                        ymean = (np.mean(m_ones)+np.mean(m_zeros))/2
                        dist = max([ymax-ymean, ymean-ymin])
                        ax.set_ylim(ymean-dist*1.1, ymean+dist*1.1)

                    # Add rate info to plot
                    self.add_rate_info([ax], m, horizontal=True)

                    # Decorate plot
                    ax.set_axisbelow(True)
                    ax.grid(linestyle='dotted')
                    ax.set_xlabel('Energy [MeV]', fontsize=12)
                    ax.set_ylabel('Model score', fontsize=12)
                    ax.set_title('Distribution of model score by lepton energy', fontsize=16)
                    leg = ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=[.5,.90], markerscale=2)
                    for lh in leg.legendHandles: 
                        lh.set_alpha(1)

            plt.savefig(self._plot_dir + models[0]._title + "_energy_score.png")
            plt.close()


    def plot_score_comparison(self, model_names=None, benchmark_names=None, shift_axes=True):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Loop over models and add data to plot
        for model, benchmark in zip(models, benchmarks):

            _, axs = plt.subplots(
                    1, 2, 
                    figsize=(18,7), 
                    gridspec_kw={'wspace': .3, 'width_ratios': [1,1]}
                )

            m_ones = model._predictions[np.where(model._truths == 1)]
            m_zeros = model._predictions[np.where(model._truths == 0)]
            b_ones = benchmark._predictions[np.where(benchmark._truths == 1)]
            b_zeros = benchmark._predictions[np.where(benchmark._truths == 0)]
            predictions = [[b_ones, m_ones], [b_zeros, m_zeros]]

            # Get correct opacity for plotting
            alpha = calculate_alpha(model._predictions)
            
            for ax, preds, label, color, is_sig in zip(axs, predictions, [self._target_label, self._background_label], [model._color, 'k'], [True, False]):
                ax.scatter(preds[0], preds[1], color=color, marker='.', alpha=alpha, label=label)

                # Decorate plot
                ax.set_axisbelow(True)
                ax.grid(linestyle='dotted')
                ax.set_xlabel(benchmark._label, fontsize=12)
                # ax.set_title('Comparison of model scores', fontsize=16)
                leg = ax.legend(fontsize=12, markerscale=2, loc='center right')
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)

                # Add rates
                self.add_rate_info([ax], benchmark, annotate=False, plot_sig=is_sig, plot_bg=not is_sig)
                self.add_rate_info([ax], model, horizontal=True, annotate=False, plot_sig=is_sig, plot_bg=not is_sig)
                
            axs[0].set_ylabel(model._label, fontsize=12)

            # Force different y-axis if the data has outliers
            if shift_axes:
                b_max = max([np.percentile(b_ones, 99.9), np.percentile(b_zeros, 99.9)])
                b_min = min([np.percentile(b_ones, .1), np.percentile(b_zeros, .1)])
                b_mean = (np.mean(b_ones)+np.mean(b_zeros))/2
                dist = max([b_max-b_mean, b_mean-b_min])

                m_max = max([np.percentile(m_ones, 99.9), np.percentile(m_zeros, 99.9)])
                m_min = min([np.percentile(m_ones, .1), np.percentile(m_zeros, .1)])
                m_mean = (np.mean(m_ones)+np.mean(m_zeros))/2
                dist = max([m_max-m_mean, m_mean-m_min])

                for ax in axs:
                    ax.set_xlim(b_mean-dist*1.1, b_mean+dist*1.1)
                    ax.set_ylim(m_mean-dist*1.1, m_mean+dist*1.1)

            plt.savefig(self._plot_dir + models[0]._title + "_model_scores.png")
            plt.close()


    def plot_score_by_position(self, model_names=None, benchmark_names=None, thresholds='inherit'):

        # TODO: Implement threshold inheritance
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Loop over models with the respective cuts
        for model, cuts in zip(models + benchmarks, thresholds):

            if model._lepton_pos is None:
                print("{}: No position found. Skipping.".format(model._name))
                continue

            fig, axses = plt.subplots(
                    2, 5, 
                    figsize=(24,9), 
                    gridspec_kw={'wspace': .3, 'hspace': .3, 'height_ratios': [1,1]}
                )

            # Get true, false and discarded rates
            pos_target_true = model._lepton_pos[np.where((model._truths == 1) & (model._predictions > cuts[0]))]
            pos_target_false = model._lepton_pos[np.where((model._truths == 1) & (model._predictions < cuts[1]))]
            pos_background_true = model._lepton_pos[np.where((model._truths == 0) & (model._predictions < cuts[1]))]
            pos_background_false = model._lepton_pos[np.where((model._truths == 0) & (model._predictions > cuts[0]))]
            pos_all_discarded = model._lepton_pos[np.where((model._predictions > cuts[1]) & (model._predictions < cuts[0]))]

            positions = [pos_target_true, pos_target_false, pos_background_true, pos_background_false, pos_all_discarded]
            labels = [self._target_label, self._target_label+' ('+self._background_label+r'$^{ID}$)', self._background_label, self._background_label+' ('+self._target_label+r'$^{ID}$)', 'Discarded']

            # Plot            
            for axs, position, label, color in zip(axses.T, positions, labels, self._color_dict['particles']):
                # Get correct opacity for plotting
                alpha = calculate_alpha(position)

                axs[0].scatter(position[:,0], position[:,1], color=color, label=label, marker='.', alpha=alpha)
                axs[1].scatter(position[:,0], position[:,2], color=color, label=label, marker='.', alpha=alpha)

                axs[0].set_xlabel('x (cm)', fontsize=12)
                axs[0].set_ylabel('y (cm)', fontsize=12)
                axs[1].set_xlabel('x (cm)', fontsize=12)
                axs[1].set_ylabel('z (cm)', fontsize=12)

                leg = axs[0].legend()
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)

            fig.suptitle('Event position distributions', fontsize=16)

            plt.savefig(self._plot_dir + model._title + "_scores_by_position.png")
            plt.close()


    def plot_score_ratio_by_distance(self, model_names=None, benchmark_names=None, thresholds='inherit', bins=20):

        # TODO: Implement threshold inheritance
        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        if thresholds != 'inherit':
            thresholds = np.array(thresholds).reshape(len(models), -1, 2)
        if len(bins) == 2:
            bins = np.linspace(bins[0], bins[1], 21)

        # Loop over models with the respective cuts
        for model, benchmark, cuts_list in zip(models, benchmarks, thresholds):
            if model._lepton_pos is None:
                print("{}: No position found. Skipping.".format(model._name))
                continue

            else:
                _, axses = plt.subplots(
                        2, 2, 
                        sharex=True,
                        figsize=(24,9), 
                        gridspec_kw={'wspace': .3, 'hspace': .05, 'height_ratios': [3,1]}
                    )

                for m, axs, cuts in zip([model, benchmark], axses.T, cuts_list):

                    # Get true and false rates
                    pos_target_true = m._lepton_pos[np.where((m._truths == 1) & (m._predictions > cuts[0]))]
                    pos_target_false = m._lepton_pos[np.where((m._truths == 1) & (m._predictions < cuts[1]))]
                    pos_background_true = m._lepton_pos[np.where((m._truths == 0) & (m._predictions < cuts[1]))]
                    pos_background_false = m._lepton_pos[np.where((m._truths == 0) & (m._predictions > cuts[0]))]

                    positions_list = [[pos_target_true, pos_target_false], [pos_background_true, pos_background_false]]
                    labels_list = [[self._target_label, self._target_label+' ('+self._background_label+r'$^{ID}$)'], [self._background_label, self._background_label+' ('+self._target_label+r'$^{ID}$)']]

                    # Plot            
                    for positions, labels, colors, color_comp in zip(positions_list, labels_list, np.array(self._color_dict['particles'][:4]).reshape(2,-1), self._color_dict['compare']):

                        radii_true = np.sqrt( positions[0][:,0]**2 + positions[0][:,1]**2 )
                        radii_false = np.sqrt( positions[1][:,0]**2 + positions[1][:,1]**2 )

                        counts_true, bins_true = np.histogram(radii_true, bins=bins)
                        counts_false, _ = np.histogram(radii_false, bins=bins_true)

                        axs[0].stairs(counts_true, bins_true, color=colors[0], label=labels[0])
                        axs[0].stairs(counts_false, bins_true, color=colors[1], label=labels[1])

                        ratios = counts_false/counts_true
                        bin_centers = bins_true[:-1] + (bins_true[1]-bins_true[0])/2

                        axs[1].plot(bin_centers, ratios, color=color_comp, marker='.', ls='dotted', label=labels[0])

                    axs[0].grid(linestyle='dotted')
                    axs[0].legend(fontsize=12)

                    axs[0].set_title('False prediction ratio - '+m._name, fontsize=16)

                    axs[1].set_ylim(0)
                    axs[1].grid(linestyle='dotted')
                    axs[1].set_xlabel('R (cm)', fontsize=12)
                    axs[1].legend(fontsize=12)
                    
                axses[0,0].set_ylabel('Counts', fontsize=12)
                axses[1,0].set_ylabel('Ratio', fontsize=12)
                    
                plt.savefig(self._plot_dir + model._title + "_scores_by_distance.png")
                plt.close()
