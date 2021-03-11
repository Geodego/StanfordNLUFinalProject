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

if __name__ == '__main__':
    a = sys.path[0]
    pass