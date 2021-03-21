"""
This file countains the ancestors of speakers and listeners. They mainly deal with building the relevant Datasets.
"""

import torch
import torch.utils

from project.models.torch_model_base import TorchModelBase
from project.utils.utils import UNK_SYMBOL, START_SYMBOL, END_SYMBOL


class ColorDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for contextual color describers. The primary
    function of this dataset is to organize the raw data into
    batches of Tensors of the appropriate shape and type. When
    using this dataset with `torch.utils.data.DataLoader`, it is
    crucial to supply the `collate_fn` method as the argument for
    the `DataLoader.collate_fn` parameter.

    Parameters
    ----------
    color_seqs : list of lists of lists of floats, or np.array
        Dimension (m, n, p) where m is the number of examples, n is
        the number of colors in each context, and p is the length
        of the color representations.

    word_seqs : list of list of int
        Dimension m, the number of examples. The length of each
        sequence can vary.

    ex_lengths : list of int
        Dimension m. Each value gives the length of the corresponding
        word sequence in `word_seqs`.

    """

    def __init__(self, color_seqs, word_seqs, ex_lengths):
        assert len(color_seqs) == len(ex_lengths)
        assert len(color_seqs) == len(word_seqs)
        self.color_seqs = color_seqs
        self.word_seqs = word_seqs
        self.ex_lengths = ex_lengths

    @staticmethod
    def collate_fn(batch):
        """
        Function for creating batches.

        Parameter
        ---------
        batch : tuple of length 3
            Contains the `color_seqs`, `word_seqs`, and `ex_lengths`,
            all as lists or similar Python iterables. The function
            turns them into Tensors.

        Returns
        -------
        color_seqs : torch.FloatTensor.
             The shape is `(m, n, p)` where `m` is the batch_size,
             `n` is the number of colors in each context, and `p` is
             the color dimensionality.

        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where `m` is
            the batch_size and `k` is the length of the longest sequence
            in the batch.

        ex_lengths : torch.LongTensor
            The true lengths of each sequence in `word_seqs. This will
            have shape `(m, )`, where `m` is the batch_size.

        targets :  torch.LongTensor
            This is a padded sequence, dimension (m, k-1), where `m` is
            the batch_size and `k` is the length of the longest sequence
            in the batch. The targets match `word_seqs` except we drop the
            first symbol, as it is always START_SYMBOL. When the loss is
            calculated, we compare this sequence to `word_seqs` excluding
            the final character, which is always the END_SYMBOL. The result
            is that each timestep t is trained to predict the symbol
            at t+1.

        """
        color_seqs, word_seqs, ex_lengths = zip(*batch)
        # Conversion to Tensors:
        color_seqs = torch.FloatTensor(color_seqs)
        word_seqs = [torch.LongTensor(seq) for seq in word_seqs]
        ex_lengths = torch.LongTensor(ex_lengths)
        # Targets as next-word predictions:
        targets = [x[1:, ] for x in word_seqs]
        # Padding
        word_seqs = torch.nn.utils.rnn.pad_sequence(
            word_seqs, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True)
        return color_seqs, word_seqs, ex_lengths, targets

    def __len__(self):
        return len(self.color_seqs)

    def __getitem__(self, idx):
        return self.color_seqs[idx], self.word_seqs[idx], self.ex_lengths[idx]


class ColorDatasetListener(ColorDataset):
    """ColorDataset adapted for listeners."""

    def __init__(self, word_seqs, ex_lengths, color_seqs):
        super(ColorDatasetListener, self).__init__(color_seqs, word_seqs, ex_lengths)

    @staticmethod
    def collate_fn(batch):
        # reorder the batch from sentence, sentence length, colors to colors, sentence, sentence length so that it
        # can be fed to the parent class
        super_batch = [(z, x, y) for x, y, z in batch]
        color_seqs, word_seqs, ex_lengths, _ = super(ColorDatasetListener, ColorDatasetListener).collate_fn(super_batch)
        return word_seqs, ex_lengths, color_seqs

    def __getitem__(self, idx):
        return self.word_seqs[idx], self.ex_lengths[idx], self.color_seqs[idx]


class ColorAgent(TorchModelBase):

    def __init__(self, vocab,
                 embedding=None,
                 embed_dim=50,
                 hidden_dim=50,
                 freeze_embedding=False,
                 color_dim=54,
                 **base_kwargs):
        """
        Ancestor of all colors agents. The primary interface to modeling contextual colors datasets.

        Parameters
        ----------
        vocab : list of str
            This should be the vocabulary. It needs to be aligned with
            `embedding` in the sense that the ith element of vocab
            should be represented by the ith row of `embedding`.

        embedding : np.array or None
            Each row represents a word in `vocab`, as described above.

        embed_dim : int
            Dimensionality for the initial embeddings. This is ignored
            if `embedding` is not None, as a specified value there
            determines this value.

        hidden_dim : int
            Dimensionality of the hidden layer.

        freeze_embedding : bool
            If True, the embedding will be updated during training. If
            False, the embedding will be frozen. This parameter applies
            to both randomly initialized and pretrained embeddings.

        color_dim: int
        dimension of color embedding

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        vocab_size : int

        word2index : dict
            A look-up from vocab items to their indices.

        index2word : dict
            A look-up for indices to vocab items.

        output_dim : int
            Same as `vocab_size`.

        start_index : int
            Index of START_SYMBOL in `self.vocab`.

        end_index : int
            Index of END_SYMBOL in `self.vocab`.

        unk_index : int
            Index of UNK_SYMBOL in `self.vocab`.

        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        super(ColorAgent, self).__init__(**base_kwargs)
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.color_dim = color_dim
        self.freeze_embedding = freeze_embedding
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2index = dict(zip(self.vocab, range(self.vocab_size)))
        self.index2word = dict(zip(range(self.vocab_size), self.vocab))
        if embedding is None:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = self.embedding.shape[1]
        self.output_dim = self.vocab_size
        self.unk_index = self.vocab.index(UNK_SYMBOL)
        self.params += ['hidden_dim', 'embed_dim', 'embedding', 'freeze_embedding', 'color_dim']

    def build_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def build_graph(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args):
        raise NotImplementedError


class ColorSpeaker(ColorAgent):

    def __init__(self, *args, **kwargs):
        """
        Ancestor of all speakers. The primary interface to modeling contextual colors datasets adapted to speakers.
        :param args:
        :param kwargs:
        """
        super(ColorSpeaker, self).__init__(*args, **kwargs)
        # speakers need a start and end symbol for producing utterances.
        self.start_index = self.vocab.index(START_SYMBOL)
        self.end_index = self.vocab.index(END_SYMBOL)

    def build_dataset(self, color_seqs, word_seqs):
        """
        Create a dataset from a list of color contexts and
        associated utterances.

        Parameters
        ----------
        color_seqs : list of lists of color representations
            We assume that each context has the same number of colors,
            each with the same shape.

        word_seqs : list of lists of utterances
            A tokenized list of words. This method uses `self.word2index`
            to turn this into a list of lists of indices.

        Returns
        -------
        ColorDataset

        """
        self.color_dim = len(color_seqs[0][0])
        word_seqs = [[self.word2index.get(w, self.unk_index) for w in seq]
                     for seq in word_seqs]
        ex_lengths = [len(seq) for seq in word_seqs]
        return ColorDataset(color_seqs, word_seqs, ex_lengths)

    def build_graph(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args):
        raise NotImplementedError


class ColorListener(ColorAgent):

    def __init__(self, *args, **kwargs):
        """
        Ancestor of all listeners. The primary interface to modeling contextual colors datasets adapted to speakers.
        :param args:
        :param kwargs:
        """
        super(ColorListener, self).__init__(*args, **kwargs)
        # Listeners don't need a start and end symbol for producing utterances.

    def build_dataset(self, color_seqs, word_seqs):
        """
        Create a dataset from a list of color contexts and
        associated utterances. This one is adapted to the listeners, no need of start and end symbols.

        Parameters
        ----------
        color_seqs : list of lists of color representations
            We assume that each context has the same number of colors,
            each with the same shape.

        word_seqs : list of lists of utterances
            A tokenized list of words. This method uses `self.word2index`
            to turn this into a list of lists of indices.

        Returns
        -------
        ColorDataset

        """
        self.color_dim = len(color_seqs[0][0])
        word_seqs = [[self.word2index.get(w, self.unk_index) for w in seq]
                     for seq in word_seqs]
        # we don't need initial and end signal for listeners so we remove them
        word_seqs = [s[1:-1] for s in word_seqs]
        ex_lengths = [len(seq) for seq in word_seqs]
        return ColorDatasetListener(word_seqs, ex_lengths, color_seqs)

    def build_graph(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args):
        raise NotImplementedError
