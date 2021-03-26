import functools

import torch
from torch import nn


class WordAndPositionalEmbedding(nn.Module):
    r"""
    # todo: Virtex
    A :class:`~torch.nn.Module` for learned word embeddings and position
    embeddings for input tokens. Each token is mapped to a fixed dimensional
    word embedding; and corresponding positional embedding based on its index.
    These are summed together followed by layer normalization and an optional
    dropout.

    Parameters
    ----------
    vocab_size: int
        Size of token vocabulary.
    hidden_size: int
        Size of token embedding vectors.
    dropout: float, optional (default = 0.1)
        Dropout probability for final dropout applied after layer normalization.
    max_caption_length: int, optional (default = 30)
        Maximum length of input captions; this is used to create a fixed
        positional embedding lookup table.
    padding_idx: int, optional (default = 0)
        Token index of ``[PAD]`` token, word embedding for these tokens will
        be a vector of zeroes (and not trainable).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 60,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Get combined word and positional embeddings for input tokens.

        Parameters
        ----------
        tokens: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length)`` containing
            a batch of caption tokens, with values in ``[0, vocab_size)``.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length, hidden_size)``
            containing corresponding token embeddings.
        """
        position_indices = self._create_position_indices(tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens: torch.Tensor):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions


class WordandPositionalEmbeddingFromPretrained(WordAndPositionalEmbedding):
    """
    Allow the use of pretrained embeddings.
    """

    def __init__(self, embedding, freeze, device, *args, **kwargs):
        super(WordandPositionalEmbeddingFromPretrained, self).__init__(*args, **kwargs)
        embedding = torch.FloatTensor(embedding).to(device)
        self.words = nn.Embedding.from_pretrained(embedding, freeze=freeze)


class WordEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 embedding=None,
                 freeze_embedding=False):
        """
        Parent class from which is derived all RNN encoders and decoders using a word embedding.

        Parameters
        ----------
        vocab_size : int

        embed_dim : int

        embedding : np.array or None
            If `None`, a random embedding is created. If `np.array`, this
            value becomes the embedding.

        """
        super().__init__()
        self.vocab_size = vocab_size
        self.freeze_embedding = freeze_embedding
        self.embedding = self._define_embedding(
            embedding, self.vocab_size, embed_dim, self.freeze_embedding)
        self.embed_dim = self.embedding.embedding_dim

    def get_embeddings(self, word_seqs, target_colors=None):
        """
        Gets the input token representations. At present, these are
        just taken directly from `self.embedding`, but `target_colors`
        can be made available in case the user wants to subclass this
        function to append these representations to each input token.

        Parameters
        ----------
        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch.

        target_colors : torch.FloatTensor
            Dimension (m, c), where m is the number of examples and
            c is the dimensionality of the color representations.

        """
        return self.embedding(word_seqs)

    @staticmethod
    def _define_embedding(embedding, vocab_size, embed_dim, freeze_embedding):
        if embedding is None:
            emb = nn.Embedding(vocab_size, embed_dim)
            emb.weight.requires_grad = not freeze_embedding
            return emb
        else:
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding)





