from sklearn.model_selection import train_test_split
import torch
from project.utils.utils import UNK_SYMBOL
from project.data.data_file_path import COLORS_SRC_FILENAME
from project.data.readers import ColorsCorpusReader
from project.models.rnn_speaker import ColorizedInputDescriber
from project.models.transformer_based import TransformerDescriber
from project.data.tokenizers import tokenize_example, represent_color_context
from project.data.word_embedding import create_glove_embedding
from project.utils.tools import save_model, load_model_states
from project.data.data_split import get_color_split
from project.models.listener import LiteralListener
from project.utils.tools import time_calc


def set_up_data(rawcols_train, texts_train, rawcols_dev, texts_dev, tokenizer, ):
    """
    Set up data so that they can be used for training
    """
    # tokenize the texts, and get the vocabulary from seqs_train
    seqs_train = [tokenizer(s) for s in texts_train]
    seqs_dev = [tokenizer(s) for s in texts_dev]

    vocab = sorted({w for toks in seqs_train for w in toks})
    vocab += [UNK_SYMBOL]

    # use Fourier transformed representation of colors
    cols_train = [represent_color_context(colors) for colors in rawcols_train]
    cols_dev = [represent_color_context(colors) for colors in rawcols_dev]
    return cols_train, seqs_train, cols_dev, seqs_dev, vocab


def get_trained_color_speaker(corpus_word_count=None, use_glove=True, tokenizer=tokenize_example,
                              speaker=ColorizedInputDescriber, max_iter=None,
                              do_train=True, prev_split=False, split_rate=None, eta=0.001, batch_size=1024, **kwargs):
    """
    Read the corpus, featurize utterances, modify color representation and return the trained model.
    :param batch_size:
    :param eta:
    :param corpus_word_count: used to get a reduced version of the corpus including only corpus_word_count utterances.
    :param use_glove: if true use glove embedding
    :param tokenizer: function used to tokenize the text
    :param speaker: Speaker model used for describing the targe color
    :param max_iter:
    :param do_train: if falsed return untrained model
    :param prev_split: if true use train_test_split on all data
    :param split_rate: used if analysis is done on a restricted part of the training data.
    :param kwargs: additional params for the model used
    :return:
    {'model': model, 'seqs_test': seqs_test, 'colors_test': colors_test}
    """
    # get data from corpus
    # corpus_train = ColorsCorpusReader(
    #     COLORS_SRC_FILENAME,
    #     word_count=corpus_word_count,
    #     normalize_colors=True)
    # examples_train = list(corpus.read())
    # rawcols, texts = zip(*[[ex.colors, ex.contents] for ex in examples])
    #
    # # split the data
    # rawcols_train, rawcols_tests, texts_train, texts_test = train_test_split(rawcols, texts, random_state=0)
    rawcols_train, texts_train, rawcols_dev, texts_dev = get_color_split(test=False,
                                                                         corpus_word_count=corpus_word_count,
                                                                         prev_split=prev_split, split_rate=split_rate)

    cols_train, seqs_train, cols_dev, seqs_dev, vocab = set_up_data(rawcols_train=rawcols_train,
                                                                    texts_train=texts_train,
                                                                    rawcols_dev=rawcols_dev,
                                                                    texts_dev=texts_dev,
                                                                    tokenizer=tokenizer)
    # If use_glove selected we use a Glove embedding and we need to modify the vocabulary accordingly
    if use_glove:
        glove_embedding, glove_vocab = create_glove_embedding(vocab)
        vocab = glove_vocab
        embedding = glove_embedding
    else:
        embedding = None

    # call the model and train it using featurized utterances and Fourier transformed color representations
    # kwargs = dict()
    if max_iter is not None:
        kwargs['max_iter'] = max_iter
    if speaker == TransformerDescriber:
        kwargs['n_attention'] = 1
        kwargs['feedforward_size'] = 75

    model = speaker(vocab=vocab, embedding=embedding, early_stopping=True, batch_size=batch_size, eta=eta, **kwargs)
    if do_train:
        model.fit(cols_train, seqs_train)
    output = {'model': model, 'seqs_test': seqs_dev, 'colors_test': cols_dev,
              'rawcols_test': rawcols_dev, 'texts_test': texts_dev}
    return output


@time_calc
def train_and_save_speaker(model, corpus_word_count, file_name, max_iter=None, split_rate=None,
                           eta=0.001, batch_size=1024):
    """
    Train and save neural speaker if a file name is given.
    :type batch_size: object
    :param eta:
    :param split_rate:
    :param model: model to train
    :param corpus_word_count: used if we want to restrict data to a smaller part of the corpus for debugging
    :param file_name: name of the file where the model will be saved. If empty, just train the model doesn't save it.
    :param max_iter: number of epoch to use for training
    :return:
    """
    output = get_trained_color_speaker(corpus_word_count=corpus_word_count, speaker=model,
                                       max_iter=max_iter, split_rate=split_rate, eta=eta, batch_size=batch_size)
    trained_model = output['model'].encoder_decoder
    score = output['model'].evaluate(output['colors_test'], output['seqs_test'])
    print(score)
    vocab = output['model'].vocab_size
    print('vocabulary size: {}'.format(vocab))
    if file_name:
        save_model(trained_model, file_name)
    return output


@time_calc
def train_and_save_listener(corpus_word_count, file_name, use_glove=True, tokenizer=tokenize_example,
                            max_iter=None, early_stopping=True, glove_dim=50, eta=0.01, batch_size=1032,
                            optimizer='Adam'):
    """
    Train and save neural listener
    :param optimizer:
    :param batch_size:
    :param eta:
    :param glove_dim:
    :param early_stopping:
    :param corpus_word_count: used if we want to restrict data to a smaller part of the corpus for debugging
    :param file_name: name of the file where the model will be saved
    :param use_glove:
    :param tokenizer:
    :param max_iter: number of epoch to use for training
    :return:
    """
    rawcols_train, texts_train, rawcols_dev, texts_dev = get_color_split(test=False,
                                                                         corpus_word_count=corpus_word_count)
    cols_train, seqs_train, cols_dev, seqs_dev, vocab = set_up_data(rawcols_train=rawcols_train,
                                                                    texts_train=texts_train,
                                                                    rawcols_dev=rawcols_dev,
                                                                    texts_dev=texts_dev,
                                                                    tokenizer=tokenizer)

    sgd = getattr(torch.optim, optimizer)

    # If use_glove selected we use a Glove embedding and we need to modify the vocabulary accordingly
    if use_glove:
        glove_embedding, glove_vocab = create_glove_embedding(vocab, glove_dim)
        vocab = glove_vocab
        embedding = glove_embedding
    else:
        embedding = None
    model = LiteralListener(vocab=vocab, embedding=embedding, early_stopping=early_stopping,
                            max_iter=max_iter, hidden_dim=100, embed_dim=glove_dim, eta=eta, batch_size=batch_size,
                            optimizer_class=sgd)
    model.fit(cols_train, seqs_train)
    output = {'model': model, 'seqs_test': seqs_dev, 'colors_test': cols_dev,
              'rawcols_test': rawcols_dev, 'texts_test': texts_dev}
    score_training = model.evaluate(cols_train, seqs_train)
    print('\nscore training:')
    print(score_training)
    print('dev score:')
    score = model.evaluate(cols_dev, seqs_dev)
    print(score)
    trained_model = model.model
    save_model(trained_model, file_name)
    return output


if __name__ == '__main__':
    # train_and_save_speaker(model=ColorizedInputDescriber, corpus_word_count=2, file_name='', max_iter=1)
    # print('\n ColorisedInputDescriber')
    # output = train_and_save_speaker(model=ColorizedInputDescriber, corpus_word_count=None, file_name='', split_rate=0.5,
    #                                 eta=0.004,
    #                                 batch_size=32)
    #
    # print('\n TransformerDescriber')
    # train_and_save_speaker(model=TransformerDescriber, corpus_word_count=None, file_name='', split_rate=0.5,
    #                        eta=0.004, batch_size=32)

    train_and_save_listener(corpus_word_count=None, file_name="Listener_monroe_split_glove_b150-e0.2",
                            use_glove=True,
                            max_iter=1000
                            , early_stopping=False, glove_dim=100, eta=0.2, batch_size=150,
                            optimizer='Adadelta')
