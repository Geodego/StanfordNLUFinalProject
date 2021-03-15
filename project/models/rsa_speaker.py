from .torch_color_describer import Encoder, EncoderDecoder, ContextualColorDescriber
from .transformer_based import TransformerDescriber


class RSASpeaker(ContextualColorDescriber):
    """
    RSA speaker based on a litteral speaker and a litteral listener
    """

    def __init__(self, *args, literal_speaker=TransformerDescriber, num_layers=1, n_attention=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.literal_speaker = literal_speaker(num_layers=num_layers, n_attention=n_attention)
        self.literal_listener

