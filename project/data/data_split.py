from .readers import ColorsCorpusReader
from .data_file_path import COLORS_TRAIN, COLORS_DEV, COLORS_TEST, COLORS_SRC_FILENAME
from sklearn.model_selection import train_test_split

def get_color_split(test: bool = False, corpus_word_count: int = None, prev_split=False):
    """
    Split corpus in colors and utterances.
    :param test: if False returns colors_train, texts_train, colors_dev, texts_dev.
    if TRUE: returns colors_test, texts_test
    :param corpus_word_count: used to get a reduced version of the corpus including only corpus_word_count utterances.
    :param prev_split: if true use train_test_split on all data
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
        rawcols_train, rawcols_tests, texts_train, texts_test = train_test_split(rawcols, texts, random_state=0)
        return rawcols_train, texts_train, rawcols_tests, texts_test

    files = [COLORS_TEST] if test else [COLORS_TRAIN, COLORS_DEV]
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
    return output

