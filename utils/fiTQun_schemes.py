from typing import List

import numpy as np
from RPP.utils.data import query_database


fitqun_dict = {
    'v_e': 'fqe_nll',
    'v_mu': 'fqmu_nll',
}

def pred_pure(
    benchmark_file: str, pred_target: str, model_name: str, event_nos: List[int]
) -> np.ndarray:
    # Get correct pred target
    pred_target = fitqun_dict[pred_target]

    # Return the pure predicted value
    pred_query = "SELECT {}, event_no FROM {} WHERE event_no IN {}".format(
        pred_target, model_name, tuple(event_nos)
    )
    pred_data = query_database(benchmark_file, pred_query)

    return pred_data[pred_target].to_numpy()


def nllh_ratio(
    benchmark_file: str, pred_target: str, model_name: str, event_nos: List[int]
) -> np.ndarray:
    # Get correct pred target
    pred_target = fitqun_dict[pred_target]

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
    # Get correct pred target
    pred_target = fitqun_dict[pred_target]

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


def energy_approx(
    benchmark_file: str, pred_target: str, model_name: str, event_nos: List[int]
) -> np.ndarray:
    """
    E_nu = ( m^2_F - m^2_IB - m^2_l + 2m_IB E_l ) /
            ( 2 (m_IB - E_l + p_l cos theta_l) )
    """

    # Return the energy approximation
    pred_query = (
        "SELECT fqe_nll, fqmu_nll, fqe_ekin, fqmu_ekin, fqe_theta, fqmu_theta, event_no FROM {} WHERE event_no IN {}".format(
            model_name, tuple(event_nos)
        )
    )
    pred_data = query_database(benchmark_file, pred_query)

    # Do pid bool
    pid_bool = pred_data['fqmu_nll']/pred_data['fqe_nll'] > 1.01

    # Define particle masses
    m_electron = 0.510998
    m_muon = 105.658
    m_proton = 938.272

    # Calculate fixed values
    m_final = m_proton
    m_bound = m_proton - 27

    # Get fiTQun outputs
    m_lepton = np.where(pid_bool, m_electron, m_muon)
    E_lepton = np.where(
        pid_bool, pred_data['fqe_ekin'], pred_data['fqmu_ekin']
    ) + m_lepton
    theta_lepton = np.where(
        pid_bool, pred_data['fqe_theta'], pred_data['fqmu_theta']
    )

    p_lepton = np.sqrt((E_lepton**2)-(m_lepton**2))*np.cos(theta_lepton)

    # Caclulate energy 
    e_top = (2*m_bound*E_lepton) - ((m_bound**2) + (m_lepton**2) - (m_final**2))
    e_bot = 2*( m_bound - E_lepton + p_lepton )
    energy = e_top / e_bot

    return energy
