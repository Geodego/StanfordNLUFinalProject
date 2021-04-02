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

