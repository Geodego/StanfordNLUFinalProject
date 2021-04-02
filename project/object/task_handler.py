from ..data.data_file_path import STUDY_TRAIN, STUDY_DEV, STUDY_TEST, TRAIN_SPEAKER, TRAIN_LISTENER, TRAIN_HYPER, \
    STUDY_SPLIT, COLORS_SRC_FILENAME
from ..data.data_split import build_corpus_repartition, get_dev_conditions
from ..data.readers import ColorsCorpusReader
from ..data.study.database import ColorDB
from ..models.rnn_speaker import ColorizedInputDescriber
from ..models.transformer_based import TransformerDescriber
from ..models.listener import LiteralListener
from ..scripts.hyper_parameters import hyperparameters_search
from ..utils.utils import fix_random_seeds
from ..utils.model_tools import initialize_agent
from ..scripts.pretrain import train_agent_with_params
from ..utils.tools import save_model, load_model_states

import os
import pandas as pd
from typing import Tuple


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

    def get_study_context_repartition(self) -> pd.DataFrame:
        """
        Gives the repartition between close, split and far trial for each of the subsets used for the study.
        """
        repartition = pd.DataFrame(columns=['size', 'far', 'split', 'close'])
        for key, path in self.data_path.items():
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

    def hyperparameters_search(self, model_id: int, param_fixed: dict, param_grid: dict,
                               save_results: bool = True) -> Tuple[dict, float]:
        """

        :param model_id: model_id in table Models in the detabase color_deb
        :param param_fixed: fixed parameters used for the analysis {'param': value...}
        :param param_grid: parameters used for the hyperparameter search: {'param': [value1,..., valuek]}
        :param save_results: If True results are saved in Table HyperSearch
        :return:
        best params: {'param': best value...}
        best score: best score obtained from the search
        """
        model_class = self.models[model_id]
        best_params, best_score = hyperparameters_search(model_class, param_grid, param_fixed)
        item = [model_id, param_fixed, param_grid, best_params, best_score]
        if save_results:
            ColorDB().write_hyper_search(item)
        return best_params, best_score

    def initialize_optimal_agent(self, hyper_id, action, corpus_word_count=None, training_data_id=None):
        """
        Return initialized agent with optimal hyperparameters saved in ColorDB
        :param hyper_id: id in table HyperParameters
        :param action: 'train' (training on all training data), 'train_speaker' (training on 'Train Speaker' data),
        'train_listener' (training on 'Train Listener' data), 'test' (test on 'Test' data) or 'hyper'
        (hyperparameters search on 'Hyper' data)
        :param corpus_word_count: used if we want to use a reduced dataset with only the selected word count.
        :param training_data_id: when action = 'test' we need to know on which data the agent has been trained to load
        it properly.
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
        output = initialize_agent(action=action, training_data_id=training_data_id, **hyper_params)
        return output

    def train_and_save_agent(self, hyper_id: int, training_data_id: int, save_memory=False, corpus_word_count=None,
                             silent=False, save_agent=True):
        """
        Train an agent and save the calculated parameters if save_agent=True
        :param hyper_id: id in table HyperParameters in ColorDB.
        :param training_data_id: id in table DataSplit of the corpus on which the training is done.
        :param save_memory: if model is to big just train, don't do the eval to save memory
        :param corpus_word_count: used if we want to use a reduced dataset with only the selected word count.
        :param silent: indicates if we want all prints
        :param save_agent: if True the calculated parameters are saved in project/data/pretrained_models
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

    def load_trained_model(self, trained_agent_id, corpus_word_count=None, test=False):
        """
        Load a pretrained agent with dev data needed for evaluation if test=False or test data if test=True. The same
        model is returned whatever the option chosen for test.
        :param trained_agent_id: id of the agent in Table TrainedAgent
        :param corpus_word_count: used if we want to use a reduced dataset with only the selected word count.
        :param test: if True test data are returned else dev data.
        :return:
        pretrained model with organised data
        if action is not test:
        {'model': pretrained model, 'seqs_train': , 'colors_train': , 'seqs_dev': featurized text,
        'colors_dev': Fourier transformed color, 'rawcolors_dev': colors as in corpus, 'texts_dev': raw text}
        if action is test
        {'model': pretrained model, 'seqs_test': , 'colors_test': }
        """
        hyper_param_id, training_data_id = (ColorDB().read_trained_agent(trained_agent_id)[k]
                                            for k in ['hyper_param_id', 'training_data_id'])
        action = self.actions[training_data_id]
        kwargs = {}
        # in order to do test we need to provide the id of the data on which the agent as been trained in order to
        # load the proper agent
        if test:
            action = 'test'
            kwargs['training_data_id'] = training_data_id
        agent_data = self.initialize_optimal_agent(hyper_id=hyper_param_id, action=action,
                                                   corpus_word_count=corpus_word_count, **kwargs)
        agent = agent_data['model']
        file_params = "trained_agent_{}".format(trained_agent_id)
        agent.model = load_model_states(agent.model, file_params, device=agent.device)
        return agent_data

    def eval_speaker_with_listener(self, trained_id_speaker, trained_id_listener, beam_search=False, test=False):
        """
        Model evaluated as a speaker, predicted utterance are processed by the selected listener and the accuracy
        of the listener using the predicted sentences is calculated as well as the blue score. A DataFrame with
        the accuracy per condition is returned.
        :param trained_id_speaker: the id of pretrained speaker in table TrainedAgent
        :param trained_id_listener: the id of pretrained listener in table TrainedAgent
        :param beam_search: used for analysis of pragmatic speaker S1. Speaker output produced using beam search.
        :param test: if true evaluation is done on test data, otherwise it is done on dev data.
        :return:
        None if test = true else
        DataFame with listener_accuracy per condition, columns ['result'] which is the listener accuracy and
                rows ['split', 'close', 'far']
        """
        speaker_data = self.load_trained_model(trained_id_speaker, test=test)
        listener_data = self.load_trained_model(trained_id_listener)
        if test:
            speaker, colors_dev, seqs_dev = (speaker_data[k] for k in ['model', 'colors_test', 'seqs_test'])
        else:
            speaker, colors_dev, seqs_dev = (speaker_data[k] for k in ['model', 'colors_dev', 'seqs_dev'])
        listener = listener_data['model']

        if not beam_search:
            # greedy search prediction
            seqs_predicted = speaker.predict(colors_dev)
        else:
            # beam search prediction
            seqs_predicted = []
            batch_size = len(colors_dev)
            mini_batch = batch_size // 20
            last_iter = int(batch_size / mini_batch)
            if batch_size % mini_batch != 0:
                last_iter += 1
            # we split the data to make the process consume less memory
            for i in range(last_iter):
                sub_colors = colors_dev[i*mini_batch: (i+1)*mini_batch]
                sub_seqs_predicted = speaker.predict_beam_search(sub_colors)
                seqs_predicted += sub_seqs_predicted
        score = listener.evaluate(colors_dev, seqs_predicted)
        print(score)

        if not test:
            conditions = get_dev_conditions()
            df = listener.get_listener_accuracy_per_condition(colors_dev, seqs_predicted, conditions)
            return df

    def eval_model_as_listener(self, trained_id_speaker, test=False):
        """
        This is for the evaluation of L1, pragmatic listener based on S0. Human utterances are used to calculate
        the accuracy of the model. The BLEU score calculated is used for measuring the quality of the sentences
        produced and is more useful to analyse the model as a literal speaker.
        :param trained_id_speaker: the id of pretrained speaker in table TrainedAgent
        :param test: if true evaluation is done on test data, otherwise it is done on dev data.
        :return:
        None if test = true else
        DataFame with listener_accuracy per condition, columns ['result'] which is the listener accuracy and
                rows ['split', 'close', 'far']
        """
        listener_data = self.load_trained_model(trained_id_speaker, test=test)
        if test:
            # in that case test data are used
            listener, colors_test, seqs_test = (listener_data[k] for k in ['model', 'colors_test', 'seqs_test'])
            score = listener.evaluate(colors_test, seqs_test)
            print(score)
        else:
            listener, colors_dev, seqs_dev = (listener_data[k] for k in ['model', 'colors_dev', 'seqs_dev'])
            score = listener.evaluate(colors_dev, seqs_dev)
            conditions = get_dev_conditions()
            df = listener.get_listener_accuracy_per_condition(colors_dev, seqs_dev, conditions)
            print(score)
            return df

    def _get_hyper_parameters(self, hyper_id):
        """
        Get hyperparameters saved in table HyperParameters
        :param hyper_id: id in table HyperParameters
        :return:
        dictionary with parameters
        """
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












