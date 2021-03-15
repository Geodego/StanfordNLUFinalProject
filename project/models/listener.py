import torch
import numpy as np
from torch import nn
from .torch_color_describer import ContextualColorDescriber
from .torch_model_base import TorchModelBase

# todo: handle below
"""Uses code of Newman et al. 2019 CS224u"""


class CaptionEncoder(nn.Module):
    """
    Implements the Monroe et al., 2017 color caption encoder.
    The CaptionEncoder takes a list of possible colors and a
    caption and converts the caption into a distribution over
    the colors. The color with the most probability mass is
    most likely to be refered to by the caption.

    To do this, the network first embeds the caption into a
    vector space, then it runs an LSTM over the vectors.
    From the last hidden state it produces a mean vector and
    covariance matrix giving a gaussian distribution over
    the color space. This vector and matrix are used to
    score each candidate color and these scores are softmaxed.
    For more details on the scoring see Monroe et al., 2017.
    """

    def __init__(self, embed_dim, hidden_dim, vocab_size, color_dim, weight_matrix=None):
        """
        Initializes CaptionEncoder.
        This initialization is based on the released code from Monroe et al., 2017.
        Further inline comments explain how this code differs.
        Args:
            embed_dim -  the output dimension for embeddings. Should be 100.
            hidden_dim - dimension used for the hidden state of the LSTM. Should be 100.
            vocab_size - the input dimension to the embedding layer.
            color_dim - number of dimensions of the input color vectors, the mean
                vector and the covariance matrix. Usually will be 54 (if using
                color_phi_fourier with resolution 3.)

        """
        super(CaptionEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if weight_matrix is not None:
            self.embed.load_state_dict({'weight': weight_matrix})

        # todo:
        # Various notes based on Will Monroe's code
        # should initialize bias to 0: https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/listener.py#L383
        # √ he also DOESN'T use dropout for the base listener
        # also non-linearity is "leaky_rectify" - I can't implement this without rewriting lstm :(, so I'm just going
        # to hope this isn't a problem
        # √ also LSTM is bidirectional (https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/listener.py#L713)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.mean = nn.Linear(2 * hidden_dim, color_dim)
        # covariance matrix is square, so we initialize it with color_dim^2 dimensions
        # we also initialize the bias to be the identity bc that's what Will does
        covar_dim = color_dim * color_dim
        self.covariance = nn.Linear(2 * hidden_dim, covar_dim)
        self.covariance.bias.data = torch.tensor(np.eye(color_dim), dtype=torch.float).flatten()
        self.logsoftmax = nn.LogSoftmax(dim=0)

        self.color_dim = color_dim
        self.hidden_dim = hidden_dim

    def forward(self, colors, caption):
        """Turns caption into distribution over colors."""
        embeddings = self.embed(caption)
        output, (hn, cn) = self.lstm(embeddings)

        # we only care about last output (first dim is batch size)
        # here we are concatenating the the last output vector of the forward direction (at index -1)
        # and the last output vector of the first direction (at index 0)
        output = torch.cat((output[:, -1, :self.hidden_dim],
                            output[:, 0, self.hidden_dim:]), 1)

        output_mean = self.mean(output)[0]
        output_covariance = self.covariance(output)[0]
        covar_matrix = output_covariance.reshape(-1, self.color_dim)  # make it a square matrix again

        # now compute score: -(f-mu)^T Sigma (f-mu)
        output_mean = output_mean.repeat(3, 1)
        diff_from_mean = colors[0] - output_mean
        scores = torch.matmul(diff_from_mean, covar_matrix)
        scores = torch.matmul(scores, diff_from_mean.transpose(0, 1))
        scores = -torch.diag(scores)
        distribution = self.logsoftmax(scores)
        return distribution

    def init_hidden_and_context(self):
        # first 2 because one vector for each direction.
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))


class LiteralListener(TorchModelBase):
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            hidden_dim=50,
            freeze_embedding=False,
            **base_kwargs):
        """
        The primary interface to modeling contextual colors datasets.

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

    def __init__(self, vocab, *args, color_dim=54, **kwargs):
        super(LiteralListener).__init__(vocab, *args, **kwargs)
        self.color_dim = color_dim
        self.model = None

    def build_graph(self):
        self.model = CaptionEncoder(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, vocab_size=self.vocab_size,
                                    color_dim=self.color_dim)
        return self.model

    def predict(self, X):
        """
        Produces and tracks model outputs
        """
        self.model.eval()
        model_outputs = []  # np.empty([len(X), len(X[0][1])]) # (num_entries, num_colors)

        for i, feature in enumerate(X):
            caption, colors = feature
            caption = torch.tensor([caption], dtype=torch.long)
            colors = torch.tensor([colors], dtype=torch.float)
            model_output_np = self.evaluate_iter((caption, colors)).view(-1).numpy()
            model_outputs.append(model_output_np)

        return np.array(model_outputs)

    def train_iter(self, caption_tensor, color_tensor, target, criterion):
        """
        Iterates through a single training pair, querying the model, getting a loss and
        updating the parameters. (TODO: add some kind of batching to this).
        """
        input_length = caption_tensor.size(0)
        loss = 0

        model_output = self.model(color_tensor, caption_tensor)
        model_output = model_output.view(1, -1)

        loss += criterion(model_output, target)

        return loss

    def evaluate_iter(self, pair):
        """
        Same as train_iter except don't use an optimizer and gradients or anything
        like that
        """
        with torch.no_grad():
            caption_tensor, color_tensor = pair
            model_output = self.model(color_tensor, caption_tensor)

            model_output = model_output.view(1, -1)
            return model_output
