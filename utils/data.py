
import numpy as np
import pandas as pd
import sqlite3


def query_database(file, query):

    # Get truth and predictions from sqlite database
    with sqlite3.connect(file) as con:

        print('Querying database..')
        pred_data = pd.read_sql(query, con)
        pred_data.sort_values('event_no', inplace=True, ignore_index=True)
        print('Done querying!')

    return pred_data


interaction_type_dict = {
    'cc': 1,
    'nc': 2,
}

class interaction_type_cutter:

    def __init__(self, interaction_type):
        self._interaction_type = interaction_type
        self._name = interaction_type

    def cut(self, event_nos, database):
        type_query = 'SELECT event_no, interaction_type FROM truth WHERE event_no IN {}'.format(tuple(event_nos))
        type_data = query_database(database, type_query)
        
        type_data.drop(type_data[type_data['interaction_type'] != interaction_type_dict[self._interaction_type]].index, inplace=True)
        return type_data['event_no']

get_cc = interaction_type_cutter('cc')
get_nc = interaction_type_cutter('nc')


class CLSC_cutter:

    def __init__(self):
        self._name = 'CLSC'

    def cut(self, event_nos, database):
        # Return ratio of nllh
        pred_query = 'select event_no, fqe_ekin, fqmu_ekin, fqe_nll, fqmu_nll, fqe_dw, fqe_dwd, fqmu_dw, fqmu_dwd, fq_q FROM fiTQun WHERE event_no IN {}'.format(tuple(event_nos))
        pred_data = query_database(database, pred_query)

        # Get the nominal hypothesis wall distances
        pred_data['fqnom_dw'] = np.where(pred_data['fqmu_nll']/pred_data['fqe_nll']>=1.01, pred_data['fqe_dw'], pred_data['fqmu_dw'])
        pred_data['fqnom_dwd'] = np.where(pred_data['fqmu_nll']/pred_data['fqe_nll']>=1.01, pred_data['fqe_dwd'], pred_data['fqmu_dwd'])

        # Drop pred_data based on the four criteria
        pred_data.drop(pred_data[pred_data['fq_q']<=1000].index, inplace=True)
        pred_data.drop(pred_data[pred_data['fqmu_ekin']/pred_data['fqe_ekin']>=2.8].index, inplace=True)
        pred_data.drop(pred_data[pred_data['fqnom_dw']<=30].index, inplace=True)
        pred_data.drop(pred_data[pred_data['fqnom_dwd']<=250].index, inplace=True)

        return pred_data['event_no'].to_numpy()

CLSC_cut = CLSC_cutter()
