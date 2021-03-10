from .torch_color_describer import Encoder, EncoderDecoder, ContextualColorDescriber
from ..modules.textual_heads import TransformerDecoder
import torch



class TransformerDescriber(ContextualColorDescriber):
    """
    Base color describer, encoder initializes embedding, doesn't handle the colorized feature of HW4 yet
    embeddings: pretrained embeddings
    """

    def __init__(self, *args, num_layers=1, n_attention=None, feedforward_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.feedforward_size = feedforward_size
        self.n_attention = n_attention

    def build_graph(self):

        # We didn't modify the encoder, so this is
        # just copied over from the original:
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim)

        if self.n_attention is None:
            self.n_attention = max(self.hidden_dim // 64, 1)

        if self.feedforward_size is None:
            self.feedforward_size = 4 * self.hidden_dim

        kwargs = dict()
        if self.embedding is not None:
            kwargs['embedding'] = self.embedding

        decoder = TransformerDecoder(visual_feature_size=self.hidden_dim, vocab_size=self.vocab_size,
                                         hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                         attention_heads=self.n_attention, feedforward_size=self.feedforward_size,
                                         **kwargs)

        self.embed_dim = decoder.embedding.words.embedding_dim
        # Return a `TransformerEncoderDecoder` that uses Encoder and TransformerTextualHead
        output = TransformerEncoderDecoder(encoder, decoder)
        self.encoder_decoder = output
        return output


class TransformerEncoderDecoder(EncoderDecoder):

    def forward(self, color_seqs, word_seqs, seq_lengths, hidden=None):

        if hidden is None:
            hidden = self.encoder(color_seqs)
        # hidden_shape: [nm_layers*num_directions, batch_size, hidden_size]
        # default batch size set up to 1028 in TorchModelBase
        # Output dense layer to get logits:
        if seq_lengths is None and not self.training:
            # should mean we're in eval mode trying to calculate proba using the pretrained model. In that case each
            # sentence has only one token
            seq_lengths = torch.ones(word_seqs.shape[0])
        output_logits = self.decoder(visual_features=hidden, caption_tokens=word_seqs, caption_lengths=seq_lengths)
        # output_logits.shape: [batch_size, max_num_word_in_batch, vocab size]


        if self.training:
            # Drop the final element:
            output = output_logits[:, : -1, :]
            # Reshape for the sake of the loss function:
            output_caption = output.transpose(1, 2)
            return output_caption
        else:

            return output_logits, hidden