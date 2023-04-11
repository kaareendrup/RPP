from typing import List, Tuple, Optional, Any

import numpy as np
from matplotlib.axes import Axes
import pathlib
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# from RPP.data.models import ClassificationModel TODO: function/method placement


def make_plot_dir(name: str, target: str, dir: str) -> str:
    plot_dir = dir + "/plots/" + target + "_" + name + "/"

    path = pathlib.Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)

    return plot_dir


def beautify_label(label: str) -> str:
    l = list(label)
    for i, c in enumerate(l):
        if c == "v":
            if i == 0:
                if not l[i + 1].isalnum():
                    l[i] = r"\nu"
            elif i == len(l) - 1:
                if not l[i - 1].isalnum():
                    l[i] = r"\nu"
            else:
                if not l[i - 1].isalnum() and not l[i + 1].isalnum():
                    l[i] = r"\nu"

    label = "".join(l)
    return r"$%s$" % (label)


fiTQun_dict = {
    "v_u": "fqmu_nll",
    "v_e": "fqe_nll",
}


def PR_flip_outputs(
    y_true: List[int], probas_pred: List[float], pos_label: Optional[int] = None
) -> Tuple[List[float]]:
    pre, rec, thresholds = precision_recall_curve(
        y_true, probas_pred, pos_label=pos_label
    )
    return rec, pre, thresholds


def get_rates(
    x: List[float],
    y: List[float],
    thresholds: List[float],
    target_rates: Optional[List[float]],
    target_cuts: Optional[List[float]],
    curve_type: Optional[str] = "ROC",
) -> List[List[float]]:
    rates_list = []

    if target_rates is not None and target_cuts is not None:
        print("Both target cuts and rates specified. Using rates.")

    if target_rates is not None:
        for rate in target_rates:
            if curve_type == "ROC":
                diffs = abs(x - (1 - rate))
            if curve_type == "PRC":
                diffs = abs(y - rate)

            idx = np.argmin(diffs)
            rates_list.append([x[idx], y[idx], thresholds[idx]])

    elif target_cuts is not None:
        for cut in target_cuts:
            diffs = abs(thresholds - cut)
            idx = np.argmin(diffs)
            rates_list.append([x[idx], y[idx], thresholds[idx]])

    return rates_list


# def add_rates(axs: Axes, model: ClassificationModel, curve_type: Optional[str] = None) -> None:
def add_rates(axs: Axes, model: Any, curve_type: Optional[str] = None) -> None:
    curve_type = model._target_curve_type if curve_type is None else curve_type

    for i_x, i_y, ithreshold in model._performance_rates[curve_type]:
        axs[-1].scatter(i_x, i_y, color="k", s=10, zorder=3)

        if curve_type == "ROC":
            axs[-1].text(
                i_x * 1.2, i_y - 0.005, "Cut: %.4g" % ithreshold, va="top", fontsize=9
            )
            axs[-1].text(
                i_x * 1.2,
                i_y - 0.035,
                "FPR: %.2g" % (i_x * 100) + "%",
                va="top",
                fontsize=9,
            )
            axs[-1].text(
                i_x * 1.2,
                i_y - 0.065,
                "TPR: %.4g" % (i_y * 100) + "%",
                va="top",
                fontsize=9,
            )

        if curve_type == "PRC":
            axs[-1].text(
                i_x - 0.15, i_y - 0.02, "Cut: %.3g" % ithreshold, va="top", fontsize=9
            )
            axs[-1].text(
                i_x - 0.15, i_y - 0.05, "Precision: %.3g" % i_y, va="top", fontsize=9
            )
            axs[-1].text(
                i_x - 0.15, i_y - 0.08, "Recall: %.3g" % i_x, va="top", fontsize=9
            )


curve_config_dict = {
    "ROC": {
        "metric_function": roc_curve, 
        "metric_score": roc_auc_score, 
        "x_label": "FPR", 
        "y_label": "TPR",
    },
    "PR": {
        "metric_function": PR_flip_outputs,
        "metric_score": average_precision_score,
        "x_label": "Efficiency (Recall)",
        "y_label": "Purity (Precision)",
    },
}


def target_extractor(target: str) -> Tuple[str]:
    if target is not None:
        if len(target) == 1:
            target, bg = target[0], target[0]
        elif len(target) == 2:
            target, bg = target[0], target[1]
        else:
            raise SystemExit("Please specify target rates and cuts as a list of lists.")
    else:
        bg = None

    return target, bg


def calculate_alpha(data: List[float]) -> float:
    l = len(data) + 1
    alpha = min(1, l ** (-0.9) * 1000)

    return alpha


def shift_axis(
    ax: Axes,
    a: List[float],
    b: List[float],
    shift_x: Optional[bool] = False,
    shift_y: Optional[bool] = False,
) -> List:
    ax_max = max([np.percentile(a, 99.9), np.percentile(b, 99.9)])
    ax_min = min([np.percentile(a, 0.1), np.percentile(b, 0.1)])
    ax_mean = (np.mean(a) + np.mean(b)) / 2
    dist = max([ax_max - ax_mean, ax_mean - ax_min])
    ax_min, ax_max = (
        (0, 1)
        if abs(1 - max(a)) < 1e-5
        else (ax_mean - dist * 1.1, ax_mean + dist * 1.1)
    )

    if shift_x:
        ax.set_xlim(ax_min, ax_max)
    if shift_y:
        ax.set_ylim(ax_min, ax_max)

    return ax, ax_min, ax_max
