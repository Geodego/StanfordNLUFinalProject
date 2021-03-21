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
# split used by Monroe et al. 2017. It uses the following split after reorganising the data
#   train: 15665
#   dev:   15670
#   test:  15659
COLORS_MONROE_TRAIN = os.path.join(data_dir, "datasets/colors/monroe_split/train_corpus_monroe.csv")
COLORS_MONROE_DEV = os.path.join(data_dir, "datasets/colors/monroe_split/dev_corpus_monroe.csv")
COLORS_MONROE_TEST = os.path.join(data_dir, "datasets/colors/monroe_split/test_corpus_monroe.csv")

# split used for this study.
#   TRAIN_SPEAKER subset of training data used for speaker
#   TRAIN_LISTENER subset of training data used for listener
#   TRAIN_HYPER subset of training data used for optimizing hyperparameters
STUDY_TRAIN = os.path.join(data_dir, "datasets/colors/study_split/train_corpus.csv")
STUDY_DEV = os.path.join(data_dir, "datasets/colors/study_split/dev_corpus.csv")
STUDY_TEST = os.path.join(data_dir, "datasets/colors/study_split/test_corpus.csv")

TRAIN_SPEAKER = os.path.join(data_dir, "datasets/colors/study_split/train_speaker.csv")
TRAIN_LISTENER = os.path.join(data_dir, "datasets/colors/study_split/train_listener.csv")
TRAIN_HYPER = os.path.join(data_dir, "datasets/colors/study_split/train_hyper.csv")

if __name__ == '__main__':
    pass