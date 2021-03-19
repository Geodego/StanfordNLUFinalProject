import os
from ..utils.utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL
from ..utils import utils
from .data_file_path import GLOVE_HOME


def create_glove_embedding(vocab, glove_dim=50):
    # Use `utils.glove2dict` to read in the GloVe file:
    if glove_dim == 50:
        glove_base_filename = 'glove.6B.50d.txt'
    elif glove_dim == 100:
        glove_base_filename = 'glove.6B.100d.txt'
    else:
        raise Exception('glove_dim needs to be 50 or 100')
    glove_src = os.path.join(GLOVE_HOME, glove_base_filename)
    glove_dict = utils.glove2dict(glove_src)

    # Use `utils.create_pretrained_embedding` to create the embedding.
    # This function will, by default, ensure that START_TOKEN,
    # END_TOKEN, and UNK_TOKEN are included in the embedding.
    embedding, new_vocab = utils.create_pretrained_embedding(glove_dict, vocab,
                                                             required_tokens=(START_SYMBOL, END_SYMBOL, UNK_SYMBOL))
    return embedding, new_vocab