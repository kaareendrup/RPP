
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from RPP.data.models import ClassificationModel, Model
from RPP.plotters.classification_plotter import ClassificationPlotter
from RPP.utils.data import query_database
from RPP.utils.style import dark_colormap
from RPP.utils.maths.maths import rotate_polar_mean


class ClassificationSpatialPlotter(ClassificationPlotter):

    def __init__(self, 
        name: str, 
        plot_dir: str, 
        target: str, 
        background: str, 
        darkmap: Optional[ListedColormap] = dark_colormap, 
        k: Optional[float] = 0.1, 
        random_seed: Optional[int] = 42, 
        **kwargs
    ):
        
        super().__init__(name, plot_dir, target, background, **kwargs)

        self._darkmap=darkmap
        self._k = k
        self._random_seed = random_seed


    def get_good_bad_pools(
        self, 
        model: ClassificationModel, 
        benchmark: ClassificationModel, 
        colorby: str, 
        allpanels: bool, 
        random_seed: Optional[int] = None
    ) -> List:
        
        if random_seed is None:
            random_seed = self._random_seed

        n_panels = 4 if allpanels else 2

        # Calculate model differences from truth
        model_diffs = abs(model._truths - model._predictions)
        benchmark_diffs = abs(benchmark._truths - benchmark._predictions)

        # Categorize events
        model_good = model._event_nos[np.where(model_diffs < self._k)]
        model_bad = model._event_nos[np.where(model_diffs > 1-self._k)]
        benchmark_good = benchmark._event_nos[np.where(benchmark_diffs < self._k)]
        benchmark_bad = benchmark._event_nos[np.where(benchmark_diffs > 1-self._k)]

        # Distribute events by performance of each model
        pools = []
        for model_selection in [model_good, model_bad]:
            for benchmark_selection in [benchmark_bad, benchmark_good]:
                pools.append(np.intersect1d(model_selection, benchmark_selection))
        
        # Setup pools and labels
        if allpanels:
            pools[1], pools[3] = pools[3], pools[1]
            labels = ['G/B', 'B/G', 'B/B', 'G/G']
        else:
            labels = ['G/B', 'G/G']

        # Loop over panels and pools to get data for a single event
        events, features_list, truths_list = [], [], []
        for i in range(n_panels):
            pool = pools[i]
            print('Number of candidate events: {}'.format(len(pool)))

            # Check if the poll has events
            if len(pool) == 0:
                events.append(None)
                features_list.append([])                
                truths_list.append([])

            else:
                # Randomly select a single event
                RNG = np.random.default_rng(seed=random_seed)
                event = RNG.choice(pool)
                print('Selected event: {}'.format(event))

                # Get event info 
                features_query = 'SELECT event_no, fX, fY, fZ, {} FROM {} WHERE event_no == {}'.format(colorby, model._pulsemap_name, event)
                features = query_database(model._db_path, features_query)

                truths_query = 'SELECT event_no, pid, particle_sign FROM truth WHERE event_no == {}'.format(event)
                truths = query_database(model._db_path, truths_query)

                events.append(event)
                features_list.append(features) 
                truths_list.append(truths)

        vmin = min([min(features[colorby]) for features in features_list])
        vmax = max([max(features[colorby]) for features in features_list])

        return events, features_list, truths_list, labels, model_diffs, benchmark_diffs, vmin, vmax
    

    def visualise_discrepancy_plot(
        self, 
        model_names: Optional[List[str]] = None, 
        benchmark_names: Optional[List[str]] = None, 
        colorby: Optional[str] = 'fTime', 
        allpanels: Optional[bool] = False
    ):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Loop over models
        for model, benchmark in zip(models, benchmarks):

            # Initialize figure
            n_panels = 4 if allpanels else 2
            fig, axs = plt.subplots(1, n_panels, figsize=(n_panels*8, 7), subplot_kw=dict(projection='3d'))

            # Get a selection of events that fit the panels, and their event info
            events, features_list, truths_list, labels, model_diffs, benchmark_diffs, vmin, vmax = self.get_good_bad_pools(
                model, benchmark, colorby, allpanels
            )

            # Loop over panels
            for ax, event, features, truths, label in zip(axs, events, features_list, truths_list, labels):

                # Create label
                label = r'$\nu_e$' if abs(truths['pid'].to_numpy()[0]) == 12 else r'$\nu_\mu$'
                if truths['particle_sign'].to_numpy()[0] == -1:
                    label = r'$\overline{'+label[1:-1]+r'}$'
                label = label + '  #' + str(event)

                # Plot
                pnt3d=ax.scatter(features['fX'], features['fY'], features['fZ'], c=features[colorby], vmin=vmin, vmax=vmax, marker='.', s=1.5, label=label)

                ax.set_xlabel('x [cm]')
                ax.set_ylabel('y [cm]')
                ax.set_zlabel('z [cm]')

                model_diff = model_diffs[np.where(model._event_nos == event)][0]
                benchmark_diff = benchmark_diffs[np.where(benchmark._event_nos == event)][0]

                ax.set_title(model._name+': {:.3f}, '.format(model_diff) + benchmark._name+': {:.3f}'.format(benchmark_diff))
                ax.legend()

                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

            # Make colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.1, 0.007, 0.8])
            cbar=fig.colorbar(pnt3d, cax=cbar_ax)
            cbar.set_label(colorby)

            plt.savefig(self._plot_dir + model._title + '_events_3D_single_{}.png'.format(colorby))
            plt.close()


    def plot_event_displays(
        self, 
        model_names: Optional[List[str]] = None, 
        benchmark_names: Optional[List[str]] = None, 
        colorby: Optional[str] = 'fTime', 
        allpanels: Optional[bool] = False, 
        auto_rotate: Optional[bool] = False, 
        force_rotate: Optional[List[float]] = None
    ):

        # Initialize
        n_panels = 4 if allpanels else 2
        force_rotate = [None]*n_panels if force_rotate is None else force_rotate

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Loop over models
        for model, benchmark in zip(models, benchmarks):

            # Initialize figure
            fig = plt.figure(figsize=(n_panels*8, 7))

            # Get a selection of events that fit the panels, and their event info
            events, features_list, truths_list, labels, model_diffs, benchmark_diffs, vmin, vmax = self.get_good_bad_pools(
                model, benchmark, colorby, allpanels
            )

            # Loop over panels
            for i, (event, features, truths, label, force_rotation) in enumerate(zip(events, features_list, truths_list, labels, force_rotate)):

                # Create label
                label = r'$\nu_e$' if abs(truths['pid'].to_numpy()[0]) == 12 else r'$\nu_\mu$'
                if truths['particle_sign'].to_numpy()[0] == -1:
                    label = r'$\overline{'+label[1:-1]+r'}$'
                label = label + '  #' + str(event)

                # Create subplots
                ax_top = plt.subplot(3, n_panels, (0*n_panels+i+1), projection='polar')
                ax_sides = plt.subplot(3, n_panels, (1*n_panels+i+1))
                ax_bottom = plt.subplot(3, n_panels, (2*n_panels+i+1), projection='polar')

                # Convert to polar and rotate if specified
                features['r'] = np.sqrt(features['fX']**2 + features['fY']**2)
                features['phi'] = np.arctan2(features['fX'], features['fY'])

                features['phi'] = rotate_polar_mean(features['phi'], auto_rotate, force_rotation)

                # Extract sides of barrel
                feats_top = features[features['fZ'] == 549.784241]
                feats_bottom = features[features['fZ'] == -549.784241]
                feats_sides = features[abs(features['fZ']) < 549.784241]

                # Plot
                ax_top.scatter(feats_top['phi'], feats_top['r'], c=feats_top[colorby], marker='.', s=1.5, cmap=self._darkmap, vmin=vmin, vmax=vmax, label=label)
                ax_bottom.scatter(feats_bottom['phi'], feats_bottom['r'], c=feats_bottom[colorby], marker='.', s=1.5, cmap=self._darkmap, vmin=vmin, vmax=vmax)
                c_data = ax_sides.scatter(feats_sides['phi'], feats_sides['fZ'], c=feats_sides[colorby], marker='.', s=1.5, cmap=self._darkmap, vmin=vmin, vmax=vmax)

                ax_top.set_theta_zero_location('S')
                ax_bottom.set_theta_zero_location('N')
                ax_bottom.set_theta_direction('clockwise')
                ax_sides.set_xlim(-np.pi,np.pi)
                ax_sides.set_ylim(-549.784241,549.784241)

                # Remove axes and set black BG
                for ax in (ax_top, ax_bottom, ax_sides):
                    ax.set_facecolor('k')
                    ax.set_axis_off()
                    ax.add_artist(ax.patch)
                    ax.patch.set_zorder(-1)

                model_diff = model_diffs[np.where(model._event_nos == event)][0]
                benchmark_diff = benchmark_diffs[np.where(benchmark._event_nos == event)][0]

                ax_top.set_title(model._name+': {:.3f}, '.format(model_diff) + benchmark._name+': {:.3f}'.format(benchmark_diff))
                ax_top.legend(loc='upper left', bbox_to_anchor=(1.08, 1.02))

            # Make colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.1, 0.007, 0.8])
            cbar=fig.colorbar(c_data, cax=cbar_ax)
            cbar.set_label(colorby)

            plt.subplots_adjust(hspace=0)
            plt.savefig(self._plot_dir + model._title + '_event_displays_single_{}.png'.format(colorby))
            plt.close()


    def plot_several_event_displays(
        self, 
        model_names: Optional[List[str]] = None, 
        benchmark_names: Optional[List[str]] = None, 
        rows: Optional[int] = 3, 
        columns: Optional[int] = 5, 
        colorby: Optional[str] = 'fTime', 
        auto_rotate: Optional[bool] = False
    ):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Loop over models
        for model, benchmark in zip(models, benchmarks):

            fig = plt.figure(figsize=(columns*8, rows*7))

            RNG = np.random.default_rng(seed=self._random_seed)
            seeds = RNG.choice(1000, (rows, columns))

            for m in range(rows):
                for n in range(columns):

                    events, features_list, truths_list, _, _, _, _, _ = self.get_good_bad_pools(
                        model, benchmark, colorby, False, seeds[m,n]
                    )
                    event, features, truths = events[0], features_list[0], truths_list[0]

                    # Create label
                    label = r'$\nu_e$' if abs(truths['pid'].to_numpy()[0]) == 12 else r'$\nu_\mu$'
                    if truths['particle_sign'].to_numpy()[0] == -1:
                        label = r'$\overline{'+label[1:-1]+r'}$'
                    label = label + '  #' + str(event)

                    # Create subplots
                    ax_top = plt.subplot(3*rows, columns, (m*3*columns+n+0*columns+1), projection='polar')
                    ax_sides = plt.subplot(3*rows, columns, (m*3*columns+n+1*columns+1))
                    ax_bottom = plt.subplot(3*rows, columns, (m*3*columns+n+2*columns+1), projection='polar')

                    # Convert to polar and rotate if specified
                    features['r'] = np.sqrt(features['fX']**2 + features['fY']**2)
                    features['phi'] = np.arctan2(features['fX'], features['fY'])

                    features['phi'] = rotate_polar_mean(features['phi'], auto_rotate)

                    # Extract sides of barrel
                    feats_top = features[features['fZ'] == 549.784241]
                    feats_bottom = features[features['fZ'] == -549.784241]
                    feats_sides = features[abs(features['fZ']) < 549.784241]

                    # Plot
                    ax_top.scatter(feats_top['phi'], feats_top['r'], c=feats_top[colorby], marker='.', s=1.5, cmap=self._darkmap, label=label)
                    ax_bottom.scatter(feats_bottom['phi'], feats_bottom['r'], c=feats_bottom[colorby], marker='.', s=1.5, cmap=self._darkmap)
                    c_data = ax_sides.scatter(feats_sides['phi'], feats_sides['fZ'], c=feats_sides[colorby], marker='.', s=1.5, cmap=self._darkmap)

                    ax_top.set_theta_zero_location('S')
                    ax_bottom.set_theta_zero_location('N')
                    ax_bottom.set_theta_direction('clockwise')
                    ax_sides.set_xlim(-np.pi,np.pi)
                    ax_sides.set_ylim(-549.784241,549.784241)

                    ax_top.legend(loc='upper left', bbox_to_anchor=(1.08, 1.02))

                    # Remove axes and set black BG
                    for ax in (ax_top, ax_bottom, ax_sides):
                        ax.set_facecolor('k')
                        ax.set_axis_off()
                        ax.add_artist(ax.patch)
                        ax.patch.set_zorder(-1)

            # Make colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.1, 0.007, 0.8])
            cbar=fig.colorbar(c_data, cax=cbar_ax)
            cbar.set_label(colorby)
            cbar_ax.yaxis.set_ticklabels([])

            plt.subplots_adjust(hspace=0)
            plt.savefig(self._plot_dir + model._title + '_event_displays_multiple_{}.png'.format(colorby))
            plt.close()
