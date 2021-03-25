from ..data.data_file_path import STUDY_TRAIN, STUDY_DEV, STUDY_TEST, TRAIN_SPEAKER, TRAIN_LISTENER, TRAIN_HYPER, \
    STUDY_SPLIT, COLORS_SRC_FILENAME
from ..data.data_split import build_corpus_repartition
from ..data.readers import ColorsCorpusReader
from ..data.study.database import ColorDB
from ..models.rnn_speaker import ColorizedInputDescriber
from ..models.transformer_based import TransformerDescriber
from ..models.listener import LiteralListener
from ..scripts.hyper_parameters import hyperparameters_search
from ..utils.utils import fix_random_seeds

import os
import pandas as pd


class TaskHandler:
    """General object handling all the tasks required for this project"""

    def __init__(self):
        self.data_path = {'all_data': COLORS_SRC_FILENAME, 'train': STUDY_TRAIN, 'dev': STUDY_DEV, 'test': STUDY_TEST,
                          'speaker': TRAIN_SPEAKER, 'listener': TRAIN_LISTENER, 'hyper': TRAIN_HYPER}
        self.models = {1: ColorizedInputDescriber, 3: TransformerDescriber, 4: LiteralListener}
        fix_random_seeds()

    def split_data_for_study(self):
        """Split the corpus data for the study and save the relevant files."""
        if len(os.listdir(STUDY_SPLIT)) != 0:
            print("Could not save the new data split.")
            print("The folder 'datasets/color/study_split' must be empty to save new splits of the data")
            return
        path_list = [self.data_path[k] for k in ['train', 'dev', 'test', 'speaker', 'listener', 'hyper']]
        build_corpus_repartition(path_list)

    def get_study_context_repartition(self):
        """
        Gives the repartition between close, split and far trial for each of the subsets used for the study.
        """
        repartition = pd.DataFrame(columns=['size', 'far', 'split', 'close'])
        for key, path in self.data_path.items():
            if key == 'test':  # we don't touch the test data
                continue
            corpus = ColorsCorpusReader(path, normalize_colors=True)
            # split = {}
            examples = list(corpus.read())
            nber_ex = len(examples)
            repart = pd.Series([ex.condition for ex in examples]).value_counts()
            repart = repart / nber_ex
            repart['size'] = nber_ex
            repart.name = key
            repartition = repartition.append(repart)

        repartition = repartition.round(3)
        repartition = repartition.astype({'size': 'int64'})
        return repartition

    def hyperparameters_search(self, model_id, param_fixed, param_grid, save_results=True):
        model_class = self.models[model_id]
        best_params, best_score = hyperparameters_search(model_class, param_grid, param_fixed)
        item = [model_id, param_fixed, param_grid, best_params, best_score]
        if save_results:
            ColorDB().write_hyper_search(item)
        return best_params, best_score












