"""
Save here the needed path for the project data
"""
import os
from pathlib import Path
import sys


root_path = Path(sys.path[0].split('project')[0] + 'project')
data_dir = os.path.join(root_path, 'data')
GLOVE_HOME = os.path.join(data_dir, 'datasets/glove')
COLORS_SRC_FILENAME = os.path.join(data_dir, "datasets/colors/filteredCorpus.csv")

if __name__ == '__main__':
    pass