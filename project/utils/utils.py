import numpy as np
import random
import sys


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Fall 2020"


START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
UNK_SYMBOL = "$UNK"


def progress_bar(msg, verbose=True):
    """
    Simple over-writing progress bar.

    """
    if verbose:
        sys.stderr.write('\r')
        sys.stderr.write(msg)
        sys.stderr.flush()


def randvec(n=50, lower=-0.5, upper=0.5):
    """
    Returns a random vector of length `n`. `w` is ignored.

    """
    return np.array([random.uniform(lower, upper) for i in range(n)])


def fix_random_seeds(
        seed=42,
        set_system=True,
        set_torch=True,
        set_tensorflow=False,
        set_torch_cudnn=True):
    """
    Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed to be set.

    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`

    set_tensorflow : bool
        Whether to set `tf.random.set_random_seed(seed)`

    set_torch : bool
        Whether to set `torch.manual_seed(seed)`

    set_torch_cudnn: bool
        Flag for whether to enable cudnn deterministic mode.
        Note that deterministic mode can have a performance impact,
        depending on your model.
        https://pytorch.org/docs/stable/notes/randomness.html

    Notes
    -----
    The function checks that PyTorch and TensorFlow are installed
    where the user asks to set seeds for them. If they are not
    installed, the seed-setting instruction is ignored. The intention
    is to make it easier to use this function in environments that lack
    one or both of these libraries.

    Even though the random seeds are explicitly set,
    the behavior may still not be deterministic (especially when a
    GPU is enabled), due to:

    * CUDA: There are some PyTorch functions that use CUDA functions
    that can be a source of non-determinism:
    https://pytorch.org/docs/stable/notes/randomness.html

    * PYTHONHASHSEED: On Python 3.3 and greater, hash randomization is
    turned on by default. This seed could be fixed before calling the
    python interpreter (PYTHONHASHSEED=0 python test.py). However, it
    seems impossible to set it inside the python program:
    https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

    """
    # set system seed
    if set_system:
        np.random.seed(seed)
        random.seed(seed)

    # set torch seed
    if set_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)

    # set torch cudnn backend
    if set_torch_cudnn:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # set tf seed
    if set_tensorflow:
        try:
            from tensorflow.compat.v1 import set_random_seed as set_tf_seed
        except ImportError:
            from tensorflow.random import set_seed as set_tf_seed
        except ImportError:
            pass
        else:
            set_tf_seed(seed)


def glove2dict(src_filename):
    """
    GloVe vectors file reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors as `np.array`.

    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


def create_pretrained_embedding(
        lookup, vocab, required_tokens=('$UNK', "<s>", "</s>")):
    """
    Create an embedding matrix from a lookup and a specified vocab.
    Words from `vocab` that are not in `lookup` are given random
    representations.

    Parameters
    ----------
    lookup : dict
        Must map words to their vector representations.

    vocab : list of str
        Words to create embeddings for.

    required_tokens : tuple of str
        Tokens that must have embeddings. If they are not available
        in the look-up, they will be given random representations.

    Returns
    -------
    np.array, list
        The np.array is an embedding for `vocab` and the `list` is
        the potentially expanded version of `vocab` that came in.

    """
    dim = len(next(iter(lookup.values())))
    embedding = np.array([lookup.get(w, randvec(dim)) for w in vocab])
    for tok in required_tokens:
        if tok not in vocab:
            vocab.append(tok)
            embedding = np.vstack((embedding, randvec(dim)))
    return embedding, vocab

