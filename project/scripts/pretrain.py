from project.models.rnn_speaker import ColorizedInputDescriber
from project.models.transformer_based import TransformerDescriber
from project.utils.tools import save_model
from project.utils.model_tools import initialize_agent
from project.models.listener import LiteralListener
from project.utils.tools import time_calc


@time_calc
def train_and_save_speaker(model, file_name, corpus_word_count=None,
                           max_iter=None, split_rate=None, prev_split=False, eta=0.001, batch_size=1024,
                           glove_dim=50, n_attention=1, feed_forward_size=75, optimizer='Adam',
                           early_stopping=False, **kwargs):
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
    :return:
    """
    output = initialize_agent(agent=model, corpus_word_count=corpus_word_count, eta=eta,
                              batch_size=batch_size, glove_dim=glove_dim, prev_split=prev_split,
                              split_rate=split_rate, max_iter=max_iter, n_attention=n_attention,
                              early_stopping=early_stopping,
                              feed_forward_size=feed_forward_size, optimizer=optimizer, **kwargs)
    model, colors_train, seqs_train = output['model'], output['colors_train'], output['seqs_train']
    model.fit(colors_train, seqs_train)

    trained_model = output['model'].model
    score_training = model.evaluate(output['colors_train'], output['seqs_train'])
    print('\nscore training:')
    print(score_training)
    print('dev score:')
    score = model.evaluate(output['colors_dev'], output['seqs_dev'])
    print(score)
    vocab = model.vocab_size
    print('vocabulary size: {}'.format(vocab))
    if file_name:
        save_model(trained_model, file_name)
    return output


@time_calc
def train_and_save_listener(corpus_word_count, file_name, max_iter=None, early_stopping=False, split_rate=None,
                            glove_dim=50, eta=0.01, batch_size=1032, optimizer='Adam', **kwargs):
    """
    Train and save neural listener
    :param split_rate:
    :param optimizer:
    :param batch_size:
    :param eta:
    :param glove_dim:
    :param early_stopping:
    :param corpus_word_count: used if we want to restrict data to a smaller part of the corpus for debugging
    :param file_name: name of the file where the model will be saved
    :param max_iter: number of epoch to use for training
    :return:
    """
    output = initialize_agent(agent=LiteralListener, corpus_word_count=corpus_word_count, early_stopping=early_stopping,
                              eta=eta, batch_size=batch_size, glove_dim=glove_dim,
                              split_rate=split_rate, max_iter=max_iter, optimizer=optimizer, **kwargs)

    #rawcols_train, texts_train, rawcols_dev, texts_dev = get_color_split(test=False,
    #                                                                     corpus_word_count=corpus_word_count)
    #cols_train, seqs_train, cols_dev, seqs_dev, vocab = set_up_data(rawcols_train=rawcols_train,
    #                                                                texts_train=texts_train,
    #                                                                rawcols_dev=rawcols_dev,
    #                                                                texts_dev=texts_dev,
    #                                                                tokenizer=tokenizer)

    # If use_glove selected we use a Glove embedding and we need to modify the vocabulary accordingly
    #if use_glove:
    #    glove_embedding, glove_vocab = create_glove_embedding(vocab, glove_dim)
    #    vocab = glove_vocab
    #    embedding = glove_embedding
    #else:
    #    embedding = None
    #model = LiteralListener(vocab=vocab, embedding=embedding, early_stopping=early_stopping,
    #                        max_iter=max_iter, hidden_dim=100, embed_dim=glove_dim, eta=eta, batch_size=batch_size,
    #                        optimizer_class=sgd)
    model, colors_train, seqs_train = output['model'], output['colors_train'], output['seqs_train']
    model.fit(colors_train, seqs_train)
    output['model'] = model
    #model.fit(cols_train, seqs_train)
    #output = {'model': model, 'seqs_test': seqs_dev, 'colors_test': cols_dev,
    #          'rawcols_test': rawcols_dev, 'texts_test': texts_dev}

    trained_model = model.model
    score_training = model.evaluate(output['colors_train'], output['seqs_train'])
    print('\nscore training:')
    print(score_training)
    print('dev score:')
    score = model.evaluate(output['colors_dev'], output['seqs_dev'])
    print(score)
    save_model(trained_model, file_name)
    return output


if __name__ == '__main__':
    train_and_save_speaker(model=TransformerDescriber, corpus_word_count=None,
                           file_name='TransformerDescriber_monroe_split_2')
    exit()
    # print('\n ColorisedInputDescriber')
    # output = train_and_save_speaker(model=ColorizedInputDescriber, corpus_word_count=None, file_name='', split_rate=0.5,
    #                                 eta=0.004,
    #                                 batch_size=32)
    #
    # print('\n TransformerDescriber')

    output = train_and_save_listener(corpus_word_count=2, file_name="", max_iter=2, early_stopping=True, glove_dim=100,
                                     eta=0.2, batch_size=2048, optimizer='Adadelta')
    listener = output['model']
    train_and_save_speaker(model=TransformerDescriber, corpus_word_count=2, file_name='', split_rate=None,
                           eta=0.0005, batch_size=64, glove_dim=100, max_iter=2, early_stopping=True)
