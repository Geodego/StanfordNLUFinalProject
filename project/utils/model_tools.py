import torch

from project.data.data_split import process_data
from project.data.word_embedding import load_embedding
from project.models.transformer_based import TransformerDescriber
from project.utils.tools import load_model_states


def initialize_agent(agent, action, corpus_word_count=None, eta=0.001, batch_size=1024,
                     glove_dim=None, hidden_dim=None, prev_split=None, split_rate=None, max_iter=None, n_attention=1,
                     feed_forward_size=75, early_stopping=False, optimizer='Adam', **kwargs):
    """

    :param agent:
    :param corpus_word_count:
    :param eta:
    :param batch_size:
    :param glove_dim:
    :param prev_split: if true use train_test_split on all data
    :param split_rate: used if analysis is done on a restricted part of the training data.
    :param max_iter:
    :param n_attention: for transformers
    :param feed_forward_size: for transformers
    :param optimizer:
    :param early_stopping:
    :param action: 'train', 'hyper' or 'test'
    :param kwargs: additional params for the model used
    :return:
    Initialized model with organised data
    if action is not test:
    {'model': initialized model, 'seqs_train': , 'colors_train': , 'seqs_dev': featurized text,
    'colors_dev': Fourier transformed color, 'rawcolors_dev': colors as in corpus, 'texts_dev': raw text}
    if action is test
    {'model': initialized model, 'seqs_test': , 'colors_test': }
    """
    if action != 'test':
        colors_train, seqs_train, colors_dev, seqs_dev, rawcolors_dev, texts_dev, train_vocab = process_data(
            action=action,
            corpus_word_count=corpus_word_count,
            prev_split=prev_split,
            split_rate=split_rate)
    else:
        colors_test, seqs_test, train_vocab = process_data(
            action=action,
            corpus_word_count=corpus_word_count,
            prev_split=prev_split,
            split_rate=split_rate)

    # If use_glove selected we use a Glove embedding and we need to modify the vocabulary accordingly
    embedding, vocab = load_embedding(glove_dim=glove_dim, vocab=train_vocab)

    sgd = getattr(torch.optim, optimizer)
    if max_iter is not None:
        kwargs['max_iter'] = max_iter
    if hidden_dim is not None:
        kwargs['hidden_dim'] = hidden_dim
    if agent == TransformerDescriber:
        kwargs['n_attention'] = n_attention
        kwargs['feedforward_size'] = feed_forward_size
        if embedding is not None:
            # makes sure the Transformer is built using the proper word embedding dim
            kwargs['hidden_dim'] = embedding.shape[1]

    model = agent(vocab=vocab, embedding=embedding, early_stopping=early_stopping, batch_size=batch_size, eta=eta,
                  optimizer_class=sgd, **kwargs)
    if action != 'test':
        output1 = {'model': model, 'seqs_train': seqs_train, 'colors_train': colors_train,
                   'seqs_dev': seqs_dev, 'colors_dev': colors_dev, 'rawcolors_dev': rawcolors_dev,
                   'texts_dev': texts_dev}
    elif action == 'test':
        output1 = {'model': model, 'seqs_test': seqs_test, 'colors_test': colors_test}

    return output1


def load_pretrained_agent(agent, file_params, corpus_word_count=None, glove_dim=100, prev_split=None, split_rate=None,
                          n_attention=1, feed_forward_size=75, test=False, **kwargs):
    """


    :param test:
    :param file_params:
    :param agent: class of the model which decoder encoder params were saved in file_params
    :param corpus_word_count:
    :param glove_dim:
    :param prev_split: if true use train_test_split on all data
    :param split_rate: used if analysis is done on a restricted part of the training data.
    :param n_attention: for transformers
    :param feed_forward_size: for transformers
    :param kwargs: additional params for the model used
    :return:
    Pretrained model with organised data
    {'model': initialized model, 'seqs_train': , 'colors_train': , 'seqs_test': featurized text,
    'colors_test': Fourier transformed color, 'rawcolors_test': colors as in corpus, 'texts_test': raw text}
    if test is False 'seqs_test', 'colors_test', 'rawcolors_test' and 'texts_test' are from dev data
    """
    output_dic = initialize_agent(agent=agent, corpus_word_count=corpus_word_count, glove_dim=glove_dim,
                                  prev_split=prev_split, split_rate=split_rate, n_attention=n_attention,
                                  feed_forward_size=feed_forward_size, test=test)
    model = output_dic['model']
    model.model = load_model_states(model.model, file_params)
    output_dic['model'] = model
    return output_dic
