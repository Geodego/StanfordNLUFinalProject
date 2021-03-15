"""
Save here the needed path for the project data
"""
import os
from pathlib import Path
import sys
from project.utils.tools import get_directory_path


data_dir = get_directory_path('data')
GLOVE_HOME = os.path.join(data_dir, 'datasets/glove')
COLORS_SRC_FILENAME = os.path.join(data_dir, "datasets/colors/filteredCorpus.csv")
# split used by Monroe et al. 2017. It uses the following split
#   train: 15665
#   dev:   15670
#   test:  15659
COLORS_TRAIN = os.path.join(data_dir, "datasets/colors/train_corpus_monroe.csv")
COLORS_DEV = os.path.join(data_dir, "datasets/colors/dev_corpus_monroe.csv")
COLORS_TEST = os.path.join(data_dir, "datasets/colors/test_corpus_monroe.csv")

if __name__ == '__main__':
    from project.data.readers import ColorsCorpusReader
    from project.data.data_split import get_color_split
    corpus_train = ColorsCorpusReader(
        COLORS_TRAIN,
        word_count=None,
        normalize_colors=True)
    corpus_dev = ColorsCorpusReader(
        COLORS_DEV,
        word_count=None,
        normalize_colors=True)
    corpus_test = ColorsCorpusReader(
        COLORS_TEST,
        word_count=None,
        normalize_colors=True)
    print(len(list(corpus_train.read())))
    print(len(list(corpus_dev.read())))
    print(len(list(corpus_test.read())))

    colors_train, texts_train, colors_dev, texts_dev = get_color_split()
    colors_test, texts_test = get_color_split(test=True)
    pass