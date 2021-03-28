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
from ..utils.model_tools import initialize_agent, load_pretrained_agent
from ..scripts.pretrain import train_agent_with_params
from ..utils.tools import save_model, load_model_states

import os
import pandas as pd


class TaskHandler:
    """General object handling all the tasks required for this project"""

    def __init__(self):
        self.data_path = {'all_data': COLORS_SRC_FILENAME, 'train': STUDY_TRAIN, 'dev': STUDY_DEV, 'test': STUDY_TEST,
                          'speaker': TRAIN_SPEAKER, 'listener': TRAIN_LISTENER, 'hyper': TRAIN_HYPER}
        self.models = {1: ColorizedInputDescriber, 3: TransformerDescriber, 4: LiteralListener}
        self.optimizers = {1: 'Adam', 2: 'Adadelta'}
        self.actions = {2: 'test', 3: 'train', 4: 'hyper', 5: 'train_listener', 6: 'train_speaker'}
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

    def initialize_optimal_agent(self, hyper_id, action, corpus_word_count=None):
        """
        Return initialized agent with optimal hyperparameters saved in ColorDB
        :param hyper_id:
        :param action:
        :param corpus_word_count:
        :return:
        Initialized model with organised data
        if action is not test:
        {'model': initialized model, 'seqs_train': , 'colors_train': , 'seqs_dev': featurized text,
        'colors_dev': Fourier transformed color, 'rawcolors_dev': colors as in corpus, 'texts_dev': raw text}
        if action is test
        {'model': initialized model, 'seqs_test': , 'colors_test': }
        """
        hyper_params = self._get_hyper_parameters(hyper_id)
        hyper_params['corpus_word_count'] = corpus_word_count
        output = initialize_agent(action=action, **hyper_params)
        return output

    def train_and_save_agent(self, hyper_id: int, training_data_id: int, save_memory=False, corpus_word_count=None,
                             silent=False, save_agent=True):
        """

        :param hyper_id: id in table HyperParameters in ColorDB.
        :param training_data_id: id in table DataSplit of the corpus on which the training is done.
        :param save_memory: if model is to big just train, don't do the eval to save memory
        :return:
        """
        action = self.actions[training_data_id]
        print('data for training used: ' + action)
        agent_data = self.initialize_optimal_agent(hyper_id=hyper_id, action=action,
                                                   corpus_word_count=corpus_word_count)
        agent, colors_train, seqs_train, colors_dev, seqs_dev = (
            agent_data[k] for k in ['model', 'colors_train', 'seqs_train', 'colors_dev', 'seqs_dev'])
        output = train_agent_with_params(agent, colors_train, seqs_train, colors_dev, seqs_dev, save_memory, silent)

        # save the trained agent
        fields = ['accuracy', 'corpus_bleu', 'training_accuracy', 'vocab_size', 'time_calc']
        try:
            results = [output[k] for k in fields]
        except KeyError:
            # this is a listener
            output['corpus_bleu'] = None
            results = [output[k] for k in fields]

        item = [hyper_id, training_data_id] + results
        trained_agent_id = ColorDB().write_trained_agent(item)
        # save the trained agent parameters
        file_name = "trained_agent_{}".format(trained_agent_id)
        trained_model = output['model'].model
        if save_agent:
            save_model(trained_model, file_name)

    def load_trained_model(self, trained_agent_id, corpus_word_count=None):
        hyper_param_id, training_data_id = (ColorDB().read_trained_agent(trained_agent_id)[k]
                                            for k in ['hyper_param_id', 'training_data_id'])
        action = self.actions[training_data_id]
        agent_data = self.initialize_optimal_agent(hyper_id=hyper_param_id, action=action,
                                                   corpus_word_count=corpus_word_count)
        agent = agent_data['model']
        file_params = "trained_agent_{}".format(trained_agent_id)
        agent.model = load_model_states(agent.model, file_params, device=agent.device)
        return agent_data

    def eval_speaker_with_listener(self, trained_id_speaker, trained_id_listener):
        speaker_data = self.load_trained_model(trained_id_speaker)
        listener_data = self.load_trained_model(trained_id_listener)
        speaker, colors_dev, seqs_dev = (speaker_data[k] for k in ['model', 'colors_dev', 'seqs_dev'])
        listener = listener_data['model']

        # greedy search prediction
        seqs_predicted = speaker.predict(colors_dev)
        # beam search prediction
        seqs_predicted_beam = speaker.predict_beam_search(colors_dev)
        greedy_score = listener.evaluate(colors_dev, seqs_predicted)
        beam_score = listener.evaluate(colors_dev, seqs_predicted_beam)
        # todo: analyse split/close/far contexts

    def _get_hyper_parameters(self, hyper_id):
        hyper_params = ColorDB().read_hyper_parameters(hyper_id)
        model_class = self.models[hyper_params.pop('model')]
        print('model used: {}'.format(model_class))
        # modify hyper_params so that it can be used to initialize the agent and get the corresponding data
        hyper_params['hidden_dim'] = hyper_params.pop('encoder_hidden_dim')
        hyper_params.pop('decoder_hidden_dim')
        hyper_params.pop('number_epochs')
        hyper_params.pop('id')
        hyper_params['agent'] = model_class
        hyper_params['optimizer'] = self.optimizers[hyper_params['optimizer']]
        hyper_params['n_attention'] = hyper_params.pop('attention_heads')
        hyper_params['num_layers'] = hyper_params.pop('number_layers')
        hyper_params['eta'] = hyper_params.pop('learning_rate')
        hyper_params['early_stopping'] = True if hyper_params['early_stopping'] == 1 else False

        glove_dim = None if hyper_params.pop('word_embedding') == 1 else 100
        hyper_params['glove_dim'] = glove_dim
        return hyper_params












