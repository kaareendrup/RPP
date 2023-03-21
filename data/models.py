
import copy

from RPP.utils.utils import beautify_label, target_extractor


class Model:
    
    def __init__(self, model_name, database, predictions, truths, event_nos, original_truths, energy, lepton_pos, color, cut_functions=None):
        self._name = model_name
        self._predictions = predictions
        self._truths = truths
        self._event_nos = event_nos
        self._original_truths = original_truths
        self._energy = energy
        self._lepton_pos = lepton_pos
        self._color = color
        self._cut_functions = cut_functions

        self._benchmark_index = None

        self._label = beautify_label(model_name)[1:-1]
        db_name = database.split("/")[-1][:-3]

        self._db_path = database
        self._title = db_name + '_' + model_name
        self._db_name = db_name
        self._pulsemap_name = db_name.split("_")[0]


    def add_benchmark(self, index):
        self._benchmark_index = index
    

class ClassificationModel(Model):

    def __init__(self, model_name, database, predictions, truths, event_nos, original_truths, energy, lepton_pos, color, cut_functions=None, target_rates=None, target_cuts=None):

        super().__init__(model_name, database, predictions, truths, event_nos, original_truths, energy, lepton_pos, color, cut_functions)

        # Get target rates based on what is specified
        target_rates, bg_rates = target_extractor(target_rates)
        target_cuts, bg_cuts = target_extractor(target_cuts)

        self._target_rates = target_rates
        self._bg_rates = bg_rates
        self._target_cuts = target_cuts
        self._bg_cuts = bg_cuts


    def invert_results(self):
        self._predictions = 1-self._predictions
        self._truths = 1-self._truths
        self._original_truths = 1-self._original_truths


    def get_background_model(self):

        background_model = copy.deepcopy(self)
        background_model.invert_results()

        # background_model = Model(
        #     self._name,
        #     self._db_path,
        #     1-self._predictions, 
        #     1-self._truths, 
        #     self._event_nos, 
        #     1-self._original_truths,
        #     self._energy, 
        #     self._lepton_pos,
        #     self._color,
        #     self._cut_functions,
        # )
        return background_model