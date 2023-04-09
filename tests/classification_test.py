import os

import pytest
import pandas as pd

from RPP.plotters.classification_plotter import ClassificationPlotter
from RPP.utils.data import InteractionTypeCutter
import RPP.utils.constants

TEST_DATA_DIR = os.path.join(
    RPP.utils.constants.TEST_DATA_DIR, "i3", "oscNext_genie_level7_v02"
)

plot_dir = 'test_plots'
results = TEST_DATA_DIR+'/test_database_reco.csv'
db = TEST_DATA_DIR+'/test_database.db'

FlavourPlotter = ClassificationPlotter('test_model', plot_dir, 'v_e', 'v_u')

assert(FlavourPlotter._target_label == r"$\nu_e$")
assert(FlavourPlotter._background_label == r"$\nu_u$")


FlavourPlotter.add_results(results, 'test_model', db, reverse=True, cut_functions=[InteractionTypeCutter('cc')])

assert(FlavourPlotter._models_list[0]._label == r'test_model')
assert(FlavourPlotter._models_list[0]._db_name == 'test_database')

assert(FlavourPlotter._models_list[0]._predictions[0] == 1 - pd.read_csv(results)['v_u_pred'][1])
