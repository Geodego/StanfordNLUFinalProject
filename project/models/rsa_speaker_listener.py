"""
Model for a pragmatic speaker, anticipating the listener reaction to its
utterances in chosing the right describtion of colors in context
"""
import torch
from torch import nn
from .transformer_based import TransformerDescriber
from .listener import LiteralListener


class EncoderDecoderListener(nn.Module):

    def __init__(self, transformer_encoder_decoder, caption_encoder):
        """
        This class knits the `TransformerEncoderDecoder` and the 'CaptionEncoder' into a single class
        that serves as the model for `PragmaticSpeakerListener`. This is
        largely a convenience: it means that `PragmaticSpeakerListener`
        can use a single `model` argument, and it allows us to localize
        the core computations in the `forward` method of this class.

        Parameters
        ----------`Encoder`

        capttion_encoder : `Decoder`

        """
        super().__init__()
        self.encoder_decoder = transformer_encoder_decoder
        self.caption_encoder = caption_encoder

    def forward(self, color_seqs, word_seqs, seq_lengths, hidden=None):
        """

        Parameters
        ----------
        color_seqs : torch.FloatTensor
            Dimension (m, n, p), where m is the number of examples,
            n is the number of colors in each context, and p is the
            dimensionality of each color.

        word_seqs : torch.LongTensor
            Dimension (m, k), where m is the number of examples and k
            is the length of all the (padded) sequences in the batch.

        seq_lengths : torch.LongTensor or None
            The true lengths of the sequences in `word_seqs`. If this
            is None, then we are predicting new sequences, so we will
            continue predicting until we hit a maximum length or we
            generate STOP_SYMBOL.

        hidden : torch.FloatTensor or None
            The hidden representation for each of the m examples in this
            batch. If this is None, we are predicting new sequences
            and so the hidden representation is computed for each timestep
            during decoding.

        Returns
        -------
        output : torch.FloatTensor
            Dimension (m, k, c), where m is the number of examples, k
            is the length of the sequences in this batch, and c is the
            number of classes (the size of the vocabulary).

        """
        # return the logits (distribution over the vocabulary) of the sentences created by the TransformerEncoderDecoder
        # shape: batch_size, max nber of words, vocab_size
        if not self.training:
            seq_lengths = None # needed for the proper code to be executed when we're evaluating
            output_logits = self.encoder_decoder.forward(color_seqs, word_seqs, seq_lengths, hidden)
            return output_logits
        output_logits = self.encoder_decoder.forward(color_seqs, word_seqs, seq_lengths, hidden)
        # keep the most likely words
        # shape: batch_size, max nber of words
        output_captions = torch.argmax(output_logits, dim=1)
        # the listener doesn't need the start and end captions, we remove them
        input_listener = output_captions[:, 1:-1]
        new_seq_lengths = [len(s) for s in input_listener]
        # embed the produced sentences so that they can be fed to the CaptionEncoder
        output_embedded = self.encoder_decoder.decoder.embedding(input_listener)
        scores = self.caption_encoder.forward(color_seqs=color_seqs, word_seqs=output_embedded,
                                              seq_lengths=new_seq_lengths, is_embedded=True)
        return scores


class PragmaticSpeakerListener(TransformerDescriber):

    def __init__(self, listener: LiteralListener, *args, **kwargs):
        """
        This pragmatic speaker is a TransformerDescriber combined with a literal speaker
        """
        super(PragmaticSpeakerListener, self).__init__(*args, **kwargs)
        self.listener = listener
        # to avoid any confusion between the EncoderDecoder embedding and the CaptionEncoder embeddings we set
        # the captionEncoder embedding to None
        self.listener.model.embedding = None

    def build_graph(self):
        encoder_decoder = super(PragmaticSpeakerListener, self).build_graph()
        model = EncoderDecoderListener(transformer_encoder_decoder=encoder_decoder,
                                       caption_encoder=self.listener.model)
        self.model = model
        return model

    def _get_loss_from_batch(self, batch):
        """
        :param batch:
        """
        color_seqs, word_seqs, seq_lengths, targets = batch
        # for utterances we use targets where the start signal as been dropped
        batch_preds = self.model(color_seqs=color_seqs, word_seqs=targets, seq_lengths=seq_lengths)

        # The expected distribution should be [0, 0, 1] giving a 1 probability to the last color, which is by
        # construction the target color
        expected_distribution = torch.ones(self.batch_size, dtype=int) * 2

        try:
            err = self.loss(batch_preds, expected_distribution)
        except ValueError:
            # last iteration of the batch has smaller dimension than self.batch_size
            current_size = targets.shape[0]
            expected_distribution = torch.ones(current_size, dtype=int) * 2
            err = self.loss(batch_preds, expected_distribution)
        return err
