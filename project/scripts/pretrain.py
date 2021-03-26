from project.models.rnn_speaker import ColorizedInputDescriber
from project.models.transformer_based import TransformerDescriber
from project.utils.tools import save_model
from project.utils.model_tools import initialize_agent
from project.utils.tools import time_calc


@time_calc
def train_and_save_agent(model, file_name, corpus_word_count=None,
                         max_iter=None, split_rate=None, prev_split=False, eta=0.001, batch_size=1024, glove_dim=50,
                         hidden_dim=None, n_attention=1, num_layers=1, feed_forward_size=75, optimizer='Adam',
                         early_stopping=False, silent=False, **kwargs):
    """
    Train and save neural speaker if a file name is given.
    :param early_stopping:
    :param feed_forward_size:
    :param n_attention:
    :param glove_dim:
    :param prev_split:
    :type batch_size: object
    :param eta:
    :param split_rate:
    :param model: model to train
    :param corpus_word_count: used if we want to restrict data to a smaller part of the corpus for debugging
    :param file_name: name of the file where the model will be saved. If empty, just train the model doesn't save it.
    :param max_iter: number of epoch to use for training
    :param optimizer:
    :param silent: if true doesn't print
    :return:
    """
    output = initialize_agent(agent=model, corpus_word_count=corpus_word_count, eta=eta,
                              batch_size=batch_size, glove_dim=glove_dim, prev_split=prev_split,
                              split_rate=split_rate, max_iter=max_iter, n_attention=n_attention,
                              num_layers=num_layers, early_stopping=early_stopping, hidden_dim=hidden_dim,
                              feed_forward_size=feed_forward_size, optimizer=optimizer, **kwargs)
    model, colors_train, seqs_train = output['model'], output['colors_train'], output['seqs_train']
    colors_dev, seqs_dev = output['colors_dev'], output['seqs_dev']

    output_fit = train_agent_with_params(agent=model, colors_train=colors_train, seqs_train=seqs_train,
                                         colors_dev=colors_dev, seqs_dev=seqs_dev)

    output['model'] = output_fit.pop('model')
    output = {**output, **output_fit}
    trained_model = output['model'].model
    # model.fit(colors_train, seqs_train)
    #
    # trained_model = output['model'].model
    # score_training = model.evaluate(output['colors_train'], output['seqs_train'])
    # score = model.evaluate(output['colors_dev'], output['seqs_dev'])
    # try:
    #     output['accuracy'] = score['listener_accuracy']
    # except KeyError:
    #     output['accuracy'] = score['accuracy']  # this must be a listener
    # vocab = model.vocab_size
    # output['vocab_size'] = vocab
    #
    # if not silent:
    #     print('\nscore training:')
    #     print(score_training)
    #     print('dev score:')
    #     print(score)
    #     print('vocabulary size: {}'.format(vocab))
    if file_name:
        save_model(trained_model, file_name)
    return output


@time_calc
def train_agent_with_params(agent, colors_train, seqs_train, colors_dev, seqs_dev, save_memory=False, silent=False):
    """

    :param agent: agent initialized with vocabulary corresponding to training data.
    :param colors_train:
    :param seqs_train:
    :param colors_dev:
    :param seqs_dev:
    :param silent:
    :return:
    """
    agent.fit(colors_train, seqs_train)
    output = {'model': agent}
    vocab = agent.vocab_size
    output['vocab_size'] = vocab

    if save_memory:
        for k in ['accuracy', 'corpus_bleu', 'training_accuracy']:
            output[k] = None
        return output

    score_training = agent.evaluate(colors_train, seqs_train)
    score = agent.evaluate(colors_dev, seqs_dev)

    try:
        output['accuracy'] = score['listener_accuracy']
        output['corpus_bleu'] = score['corpus_bleu']
        output['training_accuracy'] = score_training['listener_accuracy']
    except KeyError:
        output['accuracy'] = score['accuracy']  # this must be a listener
        output['training_accuracy'] = score_training['accuracy']


    if not silent:
        print('\nscore training:')
        print(score_training)
        print('dev score:')
        print(score)
        print('vocabulary size: {}'.format(vocab))
    return output



# @time_calc
# def train_and_save_listener(corpus_word_count, file_name, max_iter=None, early_stopping=False, split_rate=None,
#                             glove_dim=50, eta=0.01, batch_size=1032, optimizer='Adam', **kwargs):
#     """
#     Train and save neural listener
#     :param split_rate:
#     :param optimizer:
#     :param batch_size:
#     :param eta:
#     :param glove_dim:
#     :param early_stopping:
#     :param corpus_word_count: used if we want to restrict data to a smaller part of the corpus for debugging
#     :param file_name: name of the file where the model will be saved
#     :param max_iter: number of epoch to use for training
#     :return:
#     """
#     output = initialize_agent(agent=LiteralListener, corpus_word_count=corpus_word_count, early_stopping=early_stopping,
#                               eta=eta, batch_size=batch_size, glove_dim=glove_dim,
#                               split_rate=split_rate, max_iter=max_iter, optimizer=optimizer, **kwargs)
#
#     #rawcols_train, texts_train, rawcols_dev, texts_dev = get_color_split(test=False,
#     #                                                                     corpus_word_count=corpus_word_count)
#     #cols_train, seqs_train, cols_dev, seqs_dev, vocab = set_up_data(rawcols_train=rawcols_train,
#     #                                                                texts_train=texts_train,
#     #                                                                rawcols_dev=rawcols_dev,
#     #                                                                texts_dev=texts_dev,
#     #                                                                tokenizer=tokenizer)
#
#     # If use_glove selected we use a Glove embedding and we need to modify the vocabulary accordingly
#     #if use_glove:
#     #    glove_embedding, glove_vocab = create_glove_embedding(vocab, glove_dim)
#     #    vocab = glove_vocab
#     #    embedding = glove_embedding
#     #else:
#     #    embedding = None
#     #model = LiteralListener(vocab=vocab, embedding=embedding, early_stopping=early_stopping,
#     #                        max_iter=max_iter, hidden_dim=100, embed_dim=glove_dim, eta=eta, batch_size=batch_size,
#     #                        optimizer_class=sgd)
#     model, colors_train, seqs_train = output['model'], output['colors_train'], output['seqs_train']
#     model.fit(colors_train, seqs_train)
#     output['model'] = model
#     #model.fit(cols_train, seqs_train)
#     #output = {'model': model, 'seqs_test': seqs_dev, 'colors_test': cols_dev,
#     #          'rawcols_test': rawcols_dev, 'texts_test': texts_dev}
#
#     trained_model = model.model
#     score_training = model.evaluate(output['colors_train'], output['seqs_train'])
#     print('\nscore training:')
#     print(score_training)
#     print('dev score:')
#     score = model.evaluate(output['colors_dev'], output['seqs_dev'])
#     print(score)
#     save_model(trained_model, file_name)
#     return output


if __name__ == '__main__':
    pass
    # train_and_save_agent(model=ColorizedInputDescriber, action='train', corpus_word_count=2, max_iter=2,
    #                      file_name='', hidden_dim=100)
    # exit()
    # # print('\n ColorisedInputDescriber')
    # # output = train_and_save_speaker(model=ColorizedInputDescriber, corpus_word_count=None, file_name='', split_rate=0.5,
    # #                                 eta=0.004,
    # #                                 batch_size=32)
    # #
    # # print('\n TransformerDescriber')
    #
    # output = train_and_save_listener(corpus_word_count=2, file_name="", max_iter=2, early_stopping=True, glove_dim=100,
    #                                  eta=0.2, batch_size=2048, optimizer='Adadelta')
    # listener = output['model']
    # train_and_save_agent(model=TransformerDescriber, corpus_word_count=2, file_name='', split_rate=None,
    #                      eta=0.0005, batch_size=64, glove_dim=100, max_iter=2, early_stopping=True)
