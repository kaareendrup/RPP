
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
            model_diffs = abs(model._truths - model._predictions)
            benchmark_diffs = abs(benchmark._truths - benchmark._predictions)

            model_good = model._event_nos[np.where(model_diffs < k)]
            model_bad = model._event_nos[np.where(model_diffs > 1-k)]
            benchmark_good = benchmark._event_nos[np.where(benchmark_diffs < k)]
            benchmark_bad = benchmark._event_nos[np.where(benchmark_diffs > 1-k)]

            pools = []
            for model_selection in [model_good, model_bad]:
                for benchmark_selection in [benchmark_bad, benchmark_good]:
                    pools.append(np.intersect1d(model_selection, benchmark_selection))
            
            if allpanels:
                pools[1], pools[2] = pools[2], pools[1]
                labels = ['G/B', 'B/G', 'G/G', 'B/B']
            else:
                labels = ['G/B', 'G/G']

            # Loop over panels and pools
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

                truths_query = 'SELECT event_no, pid, particle_sign FROM truth WHERE event_no == {}'.format(event)
                truths = query_database(model._db_path, truths_query)

                label = r'$\nu_e$' if abs(truths['pid'].to_numpy()[0]) == 12 else r'$\nu_\mu$'
                if truths['particle_sign'].to_numpy()[0] == -1:
                    label = r'$\overline{'+label[1:-1]+r'}$'
                label = label + '  #' + str(event)

                pnt3d=ax.scatter(features['fX'], features['fY'], features['fZ'], c=features['fTime'], marker='.', s=1.5, label=label)
                cbar=fig.colorbar(pnt3d, ax=ax, shrink=0.7, aspect=20)
                cbar.set_label('Time (?)')

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

                model_diff = model_diffs[np.where(model._event_nos == event)][0]
                benchmark_diff = benchmark_diffs[np.where(benchmark._event_nos == event)][0]

                ax.set_title(model._name+': {:.3f}, '.format(model_diff) + benchmark._name+': {:.3f}'.format(benchmark_diff))
                ax.legend()

                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

            plt.tight_layout()
            plt.savefig(self._plot_dir + model._title + '_single_events_3D.png')
            plt.close()
