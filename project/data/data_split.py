import pandas as pd
from .readers import ColorsCorpusReader
from .data_file_path import COLORS_MONROE_TRAIN, COLORS_MONROE_DEV, COLORS_MONROE_TEST, COLORS_SRC_FILENAME
from .data_file_path import STUDY_TRAIN, TRAIN_SPEAKER, TRAIN_LISTENER, TRAIN_HYPER, STUDY_DEV, STUDY_TEST
from sklearn.model_selection import train_test_split

from .tokenizers import represent_color_context, monroe_tokenizer
from ..utils.utils import UNK_SYMBOL


def get_files(action):
    if action == 'train':
        return [STUDY_TRAIN, STUDY_DEV]
    elif action == 'train_speaker':
        return [TRAIN_SPEAKER, STUDY_DEV]
    elif action == 'train_listener':
        return [TRAIN_LISTENER, STUDY_DEV]
    elif action == 'hyper':
        return [TRAIN_HYPER, STUDY_DEV]
    elif action == 'test':
        return [STUDY_TEST]


def get_color_split(action: str = 'train', corpus_word_count: int = None, prev_split=False, split_rate=None):
    """
    Split corpus in colors and utterances.
    :param action: if 'train' or 'train_speaker' or 'train_listener' returns corresponding colors_train, texts_train, colors_dev, texts_dev.
    if test: returns colors_test, texts_test
    if hyper returns colors_hyper, texts_hyper
    :param corpus_word_count: used to get a reduced version of the corpus including only corpus_word_count utterances.
    :param prev_split: if true use train_test_split on all data
    :param split_rate: used if analysis is done on a restricted part of the training data.
    :return:
    return tuple of 2 lists if action is 'test'  or tuple of 4 lists in oll other cases
    """
    # if prev_split:
    #     corpus = ColorsCorpusReader(
    #         COLORS_SRC_FILENAME,
    #         word_count=corpus_word_count,
    #         normalize_colors=True)
    #     examples = list(corpus.read())
    #     rawcols, texts = zip(*[[ex.colors, ex.contents] for ex in examples])
    #
    #     # split the data
    #     rawcols_train, rawcols_test, texts_train, texts_test = train_test_split(rawcols, texts, random_state=0)
    #     return rawcols_train, texts_train, rawcols_test, texts_test

    files = get_files(action)
    output = tuple()
    for file in files:
        # get data from corpus
        corpus = ColorsCorpusReader(
            file,
            word_count=corpus_word_count,
            normalize_colors=True)
        examples = list(corpus.read())
        rawcols, texts = zip(*[[ex.colors, ex.contents] for ex in examples])
        if output:
            output = *output, rawcols, texts
        else:
            output = rawcols, texts
    # if split_rate is given we just keep split_rate of the training data. dev data remain unchanged.
    if split_rate is not None:
        rawcols_train, texts_train, rawcols_dev, texts_dev = output
        rawcols_train, _, texts_train, _ = train_test_split(rawcols_train, texts_train,
                                                            test_size=split_rate, random_state=0)
        output = rawcols_train, texts_train, rawcols_dev, texts_dev
    return output


def set_up_data(rawcolors, texts, tokenizer):
    """

    :param rawcolorss:
    :param texts:
    :param tokenizer:
    :return:
    fourier transformed colors, tokenized sentences, vocabulary including UNK_symbol
    """

    # tokenize the texts, and get the vocabulary
    seqs = [tokenizer(s) for s in texts]

    vocab = sorted({w for toks in seqs for w in toks})
    vocab += [UNK_SYMBOL]

    # use Fourier transformed representation of colors
    colors_fourier = [represent_color_context(colors) for colors in rawcolors]

    return colors_fourier, seqs, vocab


def process_data(action: str = 'train', corpus_word_count: int = None, prev_split=False, split_rate=None,
                 tokenizer=monroe_tokenizer, training_used=None):
    """
    Get data as needed for training, eval or testing
    :param action:
    :param corpus_word_count:
    :param prev_split:
    :param split_rate:
    :param tokenizer:
    :param training_used: needed when action is 'test'. In that case we need to know the vacabulary that's been used
    to train the model.
    :return:
    if action=test:
         colors_test, seqs_test, train_vocab
    if action=train or train_listener or train_speaker: (replace if needed train by train_listener or train_speaker)
        colors_train, seqs_train, colors_dev, seqs_dev, rawcolors_dev, texts_dev, train_vocab
    if action=hyper, like for train but with training done on hyper

    """
    if action in ['train', 'train_listener', 'train_speaker', 'hyper']:
        rawcolors_train, texts_train, rawcolors_dev, texts_dev = get_color_split(action=action,
                                                                                 corpus_word_count=corpus_word_count,
                                                                                 prev_split=prev_split,
                                                                                 split_rate=split_rate)

        colors_train, seqs_train, train_vocab = set_up_data(rawcolors=rawcolors_train,
                                                            texts=texts_train,
                                                            tokenizer=tokenizer)
        colors_dev, seqs_dev, _ = set_up_data(rawcolors=rawcolors_dev, texts=texts_dev, tokenizer=tokenizer)
        return colors_train, seqs_train, colors_dev, seqs_dev, rawcolors_dev, texts_dev, train_vocab

    elif action == 'test':
        rawcolors_train, texts_train, _, _ = get_color_split(action=training_used,
                                                             corpus_word_count=corpus_word_count,
                                                             prev_split=prev_split,
                                                             split_rate=split_rate)

        colors_train, seqs_train, train_vocab = set_up_data(rawcolors=rawcolors_train,
                                                            texts=texts_train,
                                                            tokenizer=tokenizer)

        rawcolors_test, texts_test = get_color_split(action='test')
        colors_test, seqs_test, _ = set_up_data(rawcolors=rawcolors_test, texts=texts_test, tokenizer=tokenizer)
        return colors_test, seqs_test, train_vocab  # vocabulary should always be coming from training data
    elif action == 'hyper':
        rawcolors_hyper, texts_hyper = get_color_split(action='hyper')
        colors_hyper, seqs_hyper, vocab_hyper = set_up_data(rawcolors=rawcolors_hyper, texts=texts_hyper,
                                                            tokenizer=tokenizer)
        return colors_hyper, seqs_hyper, vocab_hyper


def build_corpus_repartition(path_list):
    """
    Read the corpus and built the csv files that we'll be used for this analysis.
    1 training, 1 dev and one test.
    From the training a 50% split into training_listener and training_speaker. This is for studies wanting
    to train the speaker and the listener used for evaluation on different data.
    Another file used for hyperparameters tuning will be built from training.
    :param path_list: list of path where splited corpus will be saved:
        [train, dev, test, train_speaker, rain_listener, hyper]
    :return:
    """
    df = pd.read_csv(COLORS_SRC_FILENAME)
    df_analyse, df_test = train_test_split(df, random_state=0, test_size=0.2)
    df_train, df_dev = train_test_split(df_analyse, random_state=0, test_size=0.2)
    df_speaker, df_listener = train_test_split(df_train, random_state=0, test_size=0.5)
    _, df_hyper = train_test_split(df_train, random_state=0, test_size=0.15)
    df_list = [df_train, df_dev, df_test, df_speaker, df_listener, df_hyper]
    for frame, path in zip(df_list, path_list):
        frame.to_csv(path, header=True, index=False)
