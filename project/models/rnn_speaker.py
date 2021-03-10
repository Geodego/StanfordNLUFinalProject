from .torch_color_describer import Decoder, Encoder, EncoderDecoder, ContextualColorDescriber
import torch
import torch.nn as nn


"""
*****************************************************************************************************
Speaker model used for assignment 4
*****************************************************************************************************
"""


class ColorContextDecoder(Decoder):
    def __init__(self, color_dim, *args, **kwargs):
        self.color_dim = color_dim
        super().__init__(*args, **kwargs)

        # Fix the `self.rnn` attribute:
        self.rnn = nn.GRU(
            input_size=self.embed_dim + color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)

    def get_embeddings(self, word_seqs, target_colors=None):
        """
        Gets the input token representations.
        You can assume that `target_colors` is a tensor of shape
        (m, n), where m is the length of the batch (same as
        `word_seqs.shape[0]`) and n is the dimensionality of the
        color representations the model is using. The goal is
        to attached each color vector i to each of the tokens in
        the ith sequence of (the embedded version of) `word_seqs`.

        Parameters
        ----------
        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch.

        target_colors : torch.FloatTensor
            Dimension (m, c), where m is the number of examples and
            c is the dimensionality of the color representations.
        """
        pre_embedding = self.embedding(word_seqs)
        # we need the color representation to match the tensor representation of word_seqs
        color_tensor = target_colors.unsqueeze(1)
        color_tensor = torch.repeat_interleave(color_tensor, word_seqs.shape[1], dim=1)
        embedding = torch.cat((pre_embedding, color_tensor), dim=2)
        return embedding


class ColorizedEncoderDecoder(EncoderDecoder):

    def forward(self,
            color_seqs,
            word_seqs,
            seq_lengths=None,
            hidden=None,
            targets=None):
        if hidden is None:
            hidden = self.encoder(color_seqs)

        # Extract the target colors from `color_seqs` and
        # feed them to the decoder, which already has a
        # `target_colors` keyword.

        # the target colors is the last color in each example by construction. We need to build a tensor with dim
        # (len(color_seqs), 1)
        target_color_list = [colors[2].unsqueeze(0) for colors in color_seqs]
        target_colors = torch.cat(target_color_list)
        output, hidden = self.decoder(
            word_seqs, seq_lengths=seq_lengths, hidden=hidden, target_colors=target_colors)
        if self.training:
            return output
        else:
            return output, hidden


class ColorizedInputDescriber(ContextualColorDescriber):

    def build_graph(self):

        # We didn't modify the encoder, so this is
        # just copied over from the original:
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim)

        # Use your `ColorContextDecoder`, making sure
        # to pass in all the keyword arguments coming
        # from `ColorizedInputDescriber`:

        decoder = ColorContextDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            freeze_embedding=self.freeze_embedding,
            color_dim=self.color_dim)

        self.embed_dim = decoder.embed_dim
        # Return a `ColorizedEncoderDecoder` that uses
        # your encoder and decoder:

        return ColorizedEncoderDecoder(encoder, decoder)