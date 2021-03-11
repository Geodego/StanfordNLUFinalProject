from sklearn.model_selection import train_test_split
from project.utils.utils import UNK_SYMBOL
from project.data.data_file_path import COLORS_SRC_FILENAME
from project.data.readers import ColorsCorpusReader
from project.models.rnn_speaker import ColorizedInputDescriber
from project.models.transformer_based import TransformerDescriber
from project.data.tokenizers import tokenize_example, represent_color_context
from project.data.word_embedding import create_glove_embedding
from project.utils.tools import save_model, load_model_states


def get_trained_color_model(corpus_word_count=None, use_glove=True, tokenizer=tokenize_example,
                            speaker=ColorizedInputDescriber, max_iter=None, do_train=True):
    """
    Read the corpus, featurize utterances, modify color representation and return the trained model.
    :param corpus_word_count: used to get a reduced version of the corpus including only corpus_word_count utterances.
    :param use_glove: if true use glove embedding
    :param tokenizer: function used to tokenize the text
    :param speaker: Speaker model used for describing the targe color
    :param max_iter:
    :param do_train: if falsed return untrained model
    :return:
    {'model': model, 'seqs_test': seqs_test, 'colors_test': colors_test}
    """
    # get data from corpus
    corpus = ColorsCorpusReader(
        COLORS_SRC_FILENAME,
        word_count=corpus_word_count,
        normalize_colors=True)
    examples = list(corpus.read())
    rawcols, texts = zip(*[[ex.colors, ex.contents] for ex in examples])

    # split the data
    rawcols_train, rawcols_test, texts_train, texts_test = train_test_split(rawcols, texts, random_state=0)

    # tokenize the texts, and get the vocabulary from seqs_train
    seqs_train = [tokenizer(s) for s in texts_train]
    seqs_test = [tokenizer(s) for s in texts_test]

    vocab = sorted({w for toks in seqs_train for w in toks})
    vocab += [UNK_SYMBOL]

    # If use_glove selected we use a Glove embedding and we need to modify the vocabulary accordingly
    if use_glove:
        glove_embedding, glove_vocab = create_glove_embedding(vocab)
        vocab = glove_vocab
        embedding = glove_embedding
    else:
        embedding = None

    # use Fourier transformed representation of colors
    cols_train = [represent_color_context(colors) for colors in rawcols_train]
    cols_test = [represent_color_context(colors) for colors in rawcols_test]

    # call the model and train it using featurized utterances and Fourier transformed color representations
    kwargs = dict()
    if max_iter is not None:
        kwargs['max_iter'] = max_iter
    if speaker == TransformerDescriber:
        kwargs['n_attention'] = 1
        kwargs['feedforward_size'] = 75

    model = speaker(vocab=vocab, embedding=embedding, early_stopping=True, **kwargs)
    if do_train:
        model.fit(cols_train, seqs_train)
    output = {'model': model, 'seqs_test': seqs_test, 'colors_test': cols_test,
              'rawcols_test': rawcols_test, 'texts_test': texts_test}
    return output


def train_and_save(model, corpus_word_count, file_name, max_iter=None):
    output = get_trained_color_model(corpus_word_count=corpus_word_count, speaker=model, max_iter=max_iter)
    trained_model = output['model'].encoder_decoder
    save_model(trained_model, file_name)
    return output


if __name__ == '__main__':

    speaker = ColorizedInputDescriber
    output_unfit = get_trained_color_model(corpus_word_count=2, speaker=speaker, max_iter=1)
    model_unfit = output_unfit['model']
    output = train_and_save(speaker, corpus_word_count=2, max_iter=1, file_name='abc')
    model_fitted = output['model']
    if model_unfit.encoder_decoder is None:
        model_unfit.build_graph()
    encoder_decoder_loaded = load_model_states(model_unfit.encoder_decoder, file_name='abc')
    model_loaded = model_unfit
    model_loaded.encoder_decoder = encoder_decoder_loaded
    cols_test = output['colors_test']
    color = [cols_test[0]]
    predict_unfit = model_unfit.predict(color)
    predict_fitted = model_fitted.predict(color)
    predict_loaded = model_loaded.predict(color)
    pass

