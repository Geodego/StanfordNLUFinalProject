import pandas as pd
from .readers import ColorsCorpusReader
from .data_file_path import COLORS_MONROE_TRAIN, COLORS_MONROE_DEV, COLORS_MONROE_TEST, COLORS_SRC_FILENAME
from sklearn.model_selection import train_test_split

from .tokenizers import represent_color_context, monroe_tokenizer
from ..utils.utils import UNK_SYMBOL


def get_color_split(test: bool = False, corpus_word_count: int = None, prev_split=False, split_rate=None):
    """
    Split corpus in colors and utterances.
    :param test: if False returns colors_train, texts_train, colors_dev, texts_dev.
    if TRUE: returns colors_test, texts_test
    :param corpus_word_count: used to get a reduced version of the corpus including only corpus_word_count utterances.
    :param prev_split: if true use train_test_split on all data
    :param split_rate: used if analysis is done on a restricted part of the training data.
    :return:
    return tuple of 2 lists if test is True or tuple of 4 lists if test is false
    """
    if prev_split:
        corpus = ColorsCorpusReader(
            COLORS_SRC_FILENAME,
            word_count=corpus_word_count,
            normalize_colors=True)
        examples = list(corpus.read())
        rawcols, texts = zip(*[[ex.colors, ex.contents] for ex in examples])

        # split the data
        rawcols_train, rawcols_test, texts_train, texts_test = train_test_split(rawcols, texts, random_state=0)
        return rawcols_train, texts_train, rawcols_test, texts_test

    files = [COLORS_MONROE_TEST] if test else [COLORS_MONROE_TRAIN, COLORS_MONROE_DEV]
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


def process_data(test: bool = False, corpus_word_count: int = None, prev_split=False, split_rate=None,
                 tokenizer=monroe_tokenizer):
    """
    Get data as needed for training, eval or testing
    :param test:
    :param corpus_word_count:
    :param prev_split:
    :param split_rate:
    :param tokenizer:
    :return:
    if test:
         colors_test, seqs_test, train_vocab
    else:
        colors_train, seqs_train, colors_dev, seqs_dev, rawcolors_dev, texts_dev, train_vocab
    """
    rawcolors_train, texts_train, rawcolors_dev, texts_dev = get_color_split(test=False,
                                                                             corpus_word_count=corpus_word_count,
                                                                             prev_split=prev_split,
                                                                             split_rate=split_rate)

    colors_train, seqs_train, train_vocab = set_up_data(rawcolors=rawcolors_train,
                                                        texts=texts_train,
                                                        tokenizer=tokenizer)
    if test:
        rawcolors_test, texts_test = get_color_split(test=True)
        colors_test, seqs_test, _ = set_up_data(rawcolors=rawcolors_test, texts=texts_test, tokenizer=tokenizer)
        return colors_test, seqs_test, train_vocab  # vocabulary should always be coming from training data
    else:
        colors_dev, seqs_dev, _ = set_up_data(rawcolors=rawcolors_dev, texts=texts_dev, tokenizer=tokenizer)
        return colors_train, seqs_train, colors_dev, seqs_dev, rawcolors_dev, texts_dev, train_vocab


def build_corpus_repartition():
    """
    Read the corpus and built the csv files that we'll be used for this analysis.
    1 training, 1 dev and one test.
    From the training a 50% split into training_listener and training_speaker. This is for studies wanting
    to train the speaker and the listener used for evaluation on different data.
    Another file used for hyperparameters tuning will be built from training.
    """

    df = pd.read_csv(COLORS_SRC_FILENAME)
    df_analyse, df_test = train_test_split(df, random_state=0, test_size=0.2)
    a = df_analyse['condition'].value_counts()
    b = df_test['condition'].value_counts()
    df_train, df_dev = train_test_split(df_analyse, random_state=0, test_size=0.2)
    c = df_train['condition'].value_counts()
    d = df_dev['condition'].value_counts()
    df_speaker, df_listener = train_test_split(df_train, random_state=0, test_size=0.5)
    e = df_speaker['condition'].value_counts()
    _, df_hyper = train_test_split(df_train, random_state=0, test_size=0.15)
    f = df_hyper['condition'].value_counts()
    pass
    corpus = ColorsCorpusReader(
        COLORS_SRC_FILENAME,
        word_count=None,
        normalize_colors=True)
    examples = list(corpus.read())