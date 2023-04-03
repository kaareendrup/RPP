from typing import List

import numpy as np
from RPP.utils.data import query_database


def pred_pure(
    benchmark_file: str, pred_target: str, model_name: str, event_nos: List[int]
) -> np.ndarray:
    # Return the pure predicted value
    pred_query = "SELECT {}, event_no FROM {} WHERE event_no IN {}".format(
        pred_target, model_name, tuple(event_nos)
    )
    pred_data = query_database(benchmark_file, pred_query)

    return pred_data[pred_target].to_numpy()


def nllh_ratio(
    benchmark_file: str, pred_target: str, model_name: str, event_nos: List[int]
) -> np.ndarray:
    # Return ratio of nllh
    pred_query = (
        "SELECT fqe_nll, fqmu_nll, event_no FROM {} WHERE event_no IN {}".format(
            model_name, tuple(event_nos)
        )
    )
    pred_data = query_database(benchmark_file, pred_query)

    # Get the opposite label for background
    pred_background = "fqmu_nll" if pred_target != "fqmu_nll" else "fqe_nll"

    # Get values
    target_preds = pred_data[pred_target].to_numpy()
    background_preds = pred_data[pred_background].to_numpy()

    return background_preds / target_preds


def nllh_diff(
    benchmark_file: str, pred_target: str, model_name: str, event_nos: List[int]
) -> np.ndarray:
    # Return difference in nllh
    pred_query = (
        "SELECT fqe_nll, fqmu_nll, event_no FROM {} WHERE event_no IN {}".format(
            model_name, tuple(event_nos)
        )
    )
    pred_data = query_database(benchmark_file, pred_query)

    # Get the opposite label for background
    pred_background = "fqmu_nll" if pred_target != "fqmu_nll" else "fqe_nll"

    # Get values
    target_preds = pred_data[pred_target].to_numpy()
    background_preds = pred_data[pred_background].to_numpy()

    return background_preds - target_preds
