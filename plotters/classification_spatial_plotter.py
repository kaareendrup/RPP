
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from RPP.plotters.classification_plotter import ClassificationPlotter
from RPP.utils.data import query_database


class ClassificationSpatialPlotter(ClassificationPlotter):

    def visualise_discrepancy_plot(self, pulsemap_name, model_names=None, benchmark_names=None, k=0.1, random_seed=42, allpanels=False):

        # Add the correct models and benchmarks if not supplied
        models, benchmarks = self.get_models_and_benchmarks(model_names, benchmark_names)

        # Loop over models
        for model, benchmark in zip(models, benchmarks):

            # Initialize figure
            n_panels = 4 if allpanels else 2
            fig, axs = plt.subplots(1, n_panels, figsize=(n_panels*8, 7), subplot_kw=dict(projection='3d'))

            # Get a random sample from events where abs(truth - benchmark) > k, and same for model
            model_diffs = abs(model._predictions - model._truths)
            benchmark_diffs = abs(benchmark._predictions - benchmark._truths)

            model_good = model._event_nos[np.where(model_diffs < k)]
            model_bad = model._event_nos[np.where(1 - model_diffs > 1-k)]
            benchmark_good = benchmark._event_nos[np.where(benchmark_diffs < k)]
            benchmark_bad = benchmark._event_nos[np.where(benchmark_diffs > 1-k)]

            pools = []
            for model_selection in [model_good, model_bad]:
                for benchmark_selection in [benchmark_bad, benchmark_good]:
                    pools.append(np.union1d(model_selection, benchmark_selection))
            
            if allpanels:
                pools[1], pools[2] = pools[2], pools[1]
                labels = ['G/B', 'B/G', 'G/G', 'B/B']
            else:
                labels = ['G/B', 'G/G']

            # Get 
            for i in range(n_panels):
                ax, pool, label = axs[i], pools[i], labels[i]
                print('Number of candidate events: {}'.format(len(pool)))
                if len(pool) == 0:
                    print('No candidates, skipping.')
                    continue

                # Get event info for a single event
                RNG = np.random.default_rng(seed=random_seed)
                event = RNG.choice(pool)
                print('Selected event: {}'.format(event))

                features_query = 'SELECT event_no, fX, fY, fZ, fTime FROM {} WHERE event_no == {}'.format(pulsemap_name,event)
                features = query_database(model._db_path, features_query)

                pnt3d=ax.scatter(features['fX'], features['fY'], features['fZ'], c=features['fTime'], marker='.', s=1.5)

                cbar=fig.colorbar(pnt3d, ax=ax, shrink=0.7, aspect=20)
                cbar.set_label('Time (?)')

                ax.set_xlabel('x')
                ax.set_xlabel('y')
                ax.set_xlabel('z')
                ax.set_title(label)

                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

            plt.tight_layout()
            plt.savefig(self._plot_dir + model._title + '_single_events_3D.png')
            plt.close()
