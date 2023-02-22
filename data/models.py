
from RPP.utils.utils import beautify_label

class Model:
    
    def __init__(self, model_name, db_name, predictions, truths, event_nos, original_truths, energy, lepton_pos, color, target_rates=None, bg_rates=None, target_cuts=None, bg_cuts=None):
        self._name = model_name
        self._predictions = predictions
        self._truths = truths
        self._event_nos = event_nos
        self._original_truths = original_truths
        self._energy = energy
        self._lepton_pos = lepton_pos
        self._color = color
        self._target_rates = target_rates
        self._bg_rates = bg_rates
        self._target_cuts = target_cuts
        self._bg_cuts = bg_cuts

        self._benchmark_index = None

        self._label = beautify_label(model_name)[1:-1]
        self._title = db_name + '_' + model_name
        self._db_name = db_name


    def add_benchmark(self, index):
        self._benchmark_index = index


    def get_background_model(self):

        background_model = Model(
            self._name,
            self._db_name,
            1-self._predictions, 
            1-self._truths, 
            self._event_nos, 
            1-self._original_truths,
            self._energy, 
            self._lepton_pos,
            self._color,
            self._bg_rates,
            self._target_rates,
            self._bg_cuts,
            self._target_cuts,
        )
        return background_model