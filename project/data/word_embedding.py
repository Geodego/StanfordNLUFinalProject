import os
from ..utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL
from ..utils import utils
from .data_file_path import GLOVE_HOME


def create_glove_embedding(vocab, glove_base_filename='glove.6B.50d.txt'):
    # Use `utils.glove2dict` to read in the GloVe file:
    glove_src = os.path.join(GLOVE_HOME, glove_base_filename)
    glove_dict = utils.glove2dict(glove_src)

    # Use `utils.create_pretrained_embedding` to create the embedding.
    # This function will, by default, ensure that START_TOKEN,
    # END_TOKEN, and UNK_TOKEN are included in the embedding.
    embedding, new_vocab = utils.create_pretrained_embedding(glove_dict, vocab,
                                                             required_tokens=(START_SYMBOL, END_SYMBOL, UNK_SYMBOL))
    return embedding, new_vocab