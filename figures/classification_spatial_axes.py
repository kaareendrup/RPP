from typing import List, Optional

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from RPP.data.models import ClassificationModel
from RPP.figures.classification_axes import ClassificationAxes

from RPP.utils.data import query_database
from RPP.utils.maths.maths import rotate_polar_mean

class ClassificationSpatialAxes(ClassificationAxes):

    def get_good_bad_pools(
        self,
        model: ClassificationModel,
        benchmark: ClassificationModel,
        performance: str,
        colorby: str,
        random_seed: Optional[int] = None,
    ) -> List:
        if random_seed is None:
            random_seed = self._plotter._random_seed

        # Calculate model differences from truth
        model_diffs = abs(model._truths - model._predictions)
        benchmark_diffs = abs(benchmark._truths - benchmark._predictions)

        # Categorize events
        model_good = model._event_nos[np.where(model_diffs < self._plotter._k)]
        model_bad = model._event_nos[np.where(model_diffs > 1 - self._plotter._k)]
        benchmark_good = benchmark._event_nos[np.where(benchmark_diffs < self._plotter._k)]
        benchmark_bad = benchmark._event_nos[np.where(benchmark_diffs > 1 - self._plotter._k)]

        # Distribute events by performance of each model
        pools = []
        for model_selection in [model_good, model_bad]:
            for benchmark_selection in [benchmark_bad, benchmark_good]:
                pools.append(np.intersect1d(model_selection, benchmark_selection))

        # Setup pools and labels
        pool_dict = {"gb": pools[0], "gg": pools[1], "bb": pools[2], "bg": pools[3]}
        pool = pool_dict[performance]


        print("Number of candidate events: {}".format(len(pool)))
        # Check if the poll has events
        if len(pool) == 0:
            event = None
            features, truths = [], []
            vmin, vmax = 0, 1

        else:
            # Randomly select a single event
            RNG = np.random.default_rng(seed=random_seed)
            event = RNG.choice(pool)
            print("Selected event: {}".format(event))

            # Get event info
            features_query = "SELECT event_no, fX, fY, fZ, {} FROM {} WHERE event_no == {}".format(
                colorby, model._pulsemap_name, event
            )
            features = query_database(model._db_path, features_query)

            truths_query = "SELECT event_no, pid, fSign FROM truth WHERE event_no == {}".format(
                event
            )
            truths = query_database(model._db_path, truths_query)

            vmin = min(features[colorby])
            vmax = max(features[colorby])

        return (
            event,
            features,
            truths,
            model_diffs,
            benchmark_diffs,
            vmin,
            vmax,
        )

    def visualise_discrepancy_plot(
        self,
        model: ClassificationModel,
        benchmark: Optional[ClassificationModel] = None,
        performance: Optional[str] = "gg",
        colorby: Optional[str] = "fTime",
    ):
        # Add the correct benchmarks if not supplied
        if benchmark is None:
            benchmark = self._plotter.get_benchmarks([model])[0]

        # Get a selection of events that fit the panels, and their event info
        (
            event,
            features,
            truths,
            model_diffs,
            benchmark_diffs,
            vmin,
            vmax,
        ) = self.get_good_bad_pools(model, benchmark, performance, colorby)

        if event is not None:
            # Create label
            label = (
                r"$\nu_e$"
                if abs(truths["pid"].to_numpy()[0]) == 12
                else r"$\nu_\mu$"
            )
            if truths["fSign"].to_numpy()[0] == -1:
                label = r"$\overline{" + label[1:-1] + r"}$"
            label = label + "  #" + str(event)

            # Plot
            self.set_axis_off()
            axis3d = self.inset_axes([0,0,1,1], projection="3d")
            pnt3d = axis3d.scatter(
                features["fX"],
                features["fY"],
                features["fZ"],
                c=features[colorby],
                vmin=vmin,
                vmax=vmax,
                marker=".",
                s=1.5,
                label=label,
            )

            axis3d.set_xlabel("x [cm]")
            axis3d.set_ylabel("y [cm]")
            axis3d.set_zlabel("z [cm]")

            model_diff = model_diffs[np.where(model._event_nos == event)][0]
            benchmark_diff = benchmark_diffs[
                np.where(benchmark._event_nos == event)
            ][0]

            axis3d.set_title(
                model._name
                + ": {:.3f}, ".format(model_diff)
                + benchmark._name
                + ": {:.3f}".format(benchmark_diff)
            )
            axis3d.legend()

            axis3d.xaxis.pane.fill = False
            axis3d.yaxis.pane.fill = False
            axis3d.zaxis.pane.fill = False

        else:
            print('No event found!')

        # # Make colorbar
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.92, 0.1, 0.007, 0.8])
        # cbar = fig.colorbar(pnt3d, cax=cbar_ax)
        # cbar.set_label(colorby)

    def plot_event_displays(
        self,
        model: ClassificationModel,
        benchmark: Optional[ClassificationModel] = None,
        performance: Optional[str] = "gg",
        colorby: Optional[str] = "fTime",
        auto_rotate: Optional[bool] = False,
        force_rotate: Optional[List[float]] = None,
    ):

        # Add the correct benchmarks if not supplied
        if benchmark is None:
            benchmark = self._plotter.get_benchmarks([model])[0]

        # Get a selection of events that fit the panels, and their event info
        (
            event,
            features,
            truths,
            model_diffs,
            benchmark_diffs,
            vmin,
            vmax,
        ) = self.get_good_bad_pools(model, benchmark, performance, colorby)

        # Create label
        label = (
            r"$\nu_e$"
            if abs(truths["pid"].to_numpy()[0]) == 12
            else r"$\nu_\mu$"
        )
        if truths["particle_sign"].to_numpy()[0] == -1:
            label = r"$\overline{" + label[1:-1] + r"}$"
        label = label + "  #" + str(event)

        ax_top = inset_axes(self, [0, 0/3, 1, 1/3], projection="polar")
        ax_sides = inset_axes(self, [0, 1/3, 1, 1/3])
        ax_bottom = inset_axes(self, [0, 2/3, 1, 1/3], projection="polar")

        # Convert to polar and rotate if specified
        features["r"] = np.sqrt(features["fX"] ** 2 + features["fY"] ** 2)
        features["phi"] = np.arctan2(features["fX"], features["fY"])

        features["phi"] = rotate_polar_mean(
            features["phi"], auto_rotate, force_rotate
        )

        # Extract sides of barrel
        feats_top = features[features["fZ"] == 549.784241]
        feats_bottom = features[features["fZ"] == -549.784241]
        feats_sides = features[abs(features["fZ"]) < 549.784241]

        # Plot
        ax_top.scatter(
            feats_top["phi"],
            feats_top["r"],
            c=feats_top[colorby],
            marker=".",
            s=1.5,
            cmap=self._darkmap,
            vmin=vmin,
            vmax=vmax,
            label=label,
        )
        ax_bottom.scatter(
            feats_bottom["phi"],
            feats_bottom["r"],
            c=feats_bottom[colorby],
            marker=".",
            s=1.5,
            cmap=self._darkmap,
            vmin=vmin,
            vmax=vmax,
        )
        c_data = ax_sides.scatter(
            feats_sides["phi"],
            feats_sides["fZ"],
            c=feats_sides[colorby],
            marker=".",
            s=1.5,
            cmap=self._darkmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax_top.set_theta_zero_location("S")
        ax_bottom.set_theta_zero_location("N")
        ax_bottom.set_theta_direction("clockwise")
        ax_sides.set_xlim(-np.pi, np.pi)
        ax_sides.set_ylim(-549.784241, 549.784241)

        # Remove axes and set black BG
        for ax in (ax_top, ax_bottom, ax_sides):
            ax.set_facecolor("k")
            ax.set_axis_off()
            ax.add_artist(ax.patch)
            ax.patch.set_zorder(-1)

        model_diff = model_diffs[np.where(model._event_nos == event)][0]
        benchmark_diff = benchmark_diffs[
            np.where(benchmark._event_nos == event)
        ][0]

        ax_top.set_title(
            model._name
            + ": {:.3f}, ".format(model_diff)
            + benchmark._name
            + ": {:.3f}".format(benchmark_diff)
        )
        ax_top.legend(loc="upper left", bbox_to_anchor=(1.08, 1.02))

        # # Make colorbar
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.92, 0.1, 0.007, 0.8])
        # cbar = fig.colorbar(c_data, cax=cbar_ax)
        # cbar.set_label(colorby)

    def plot_several_event_displays(
        self,
        model_names: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        rows: Optional[int] = 3,
        columns: Optional[int] = 5,
        colorby: Optional[str] = "fTime",
        auto_rotate: Optional[bool] = False,
    ):
        # Pass
        pass