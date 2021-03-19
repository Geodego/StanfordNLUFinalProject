import torch
import numpy as np
from torch import nn
from .torch_color_describer import ColorAgent
from ..modules.embedding import WordEmbedding
from .torch_model_base import TorchModelBase

# todo: handle below, template


class CaptionEncoder(WordEmbedding):
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

    def __init__(self, hidden_dim, color_dim, *args, **kwargs):
        """
        Initializes CaptionEncoder.
        This initialization is based on the released code from Monroe et al., 2017.
        Further inline comments explain how this code differs.
        Args:
            hidden_dim - dimension used for the hidden state of the LSTM. Should be 100.
            color_dim - number of dimensions of the input color vectors, the mean
                vector and the covariance matrix. Usually will be 54 (if using
                color_phi_fourier with resolution 3.)
        """
        super(CaptionEncoder, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        # todo:
        # Various notes based on Will Monroe's code
        # should initialize bias to 0:
        # https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/listener.py#L383
        # √ he also DOESN'T use dropout for the base listener
        # also non-linearity is "leaky_rectify" - I can't implement this without rewriting lstm :(, so I'm just going
        # to hope this isn't a problem
        # √ also LSTM is bidirectional :
        # https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/listener.py#L713

        self.lstm = nn.LSTM(self.embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.mean = nn.Linear(2 * hidden_dim, color_dim)
        # covariance matrix is square, so we initialize it with color_dim^2 dimensions
        # we also initialize the bias to be the identity bc that's what Will does
        covar_dim = color_dim * color_dim
        self.covariance = nn.Linear(2 * hidden_dim, covar_dim)
        self.covariance.bias.data = torch.tensor(np.eye(color_dim), dtype=torch.float).flatten()
        #self.logsoftmax = nn.LogSoftmax(dim=0)

        self.color_dim = color_dim
        self.hidden_dim = hidden_dim

    def forward(self, color_seqs, word_seqs):
        """Turns caption into distribution over colors."""
        #todo: pack sequences for performance
        batch_size = color_seqs.shape[0]
        embeddings = self.get_embeddings(word_seqs)
        output1, (hn, cn) = self.lstm(embeddings)

        # we only care about last output (first dim is batch size)
        # here we are concatenating the the last output vector of the forward direction (at index -1)
        # and the last output vector of the first direction (at index 0)
        output = torch.cat((output1[:, -1, :self.hidden_dim],
                            output1[:, 0, self.hidden_dim:]), 1)

        output_mean = self.mean(output)
        output_covariance = self.covariance(output)
        covar_matrix = output_covariance.reshape(batch_size, -1, self.color_dim)  # make it a square matrix again

        # now compute score: -(f-mu)^T Sigma (f-mu)
        output_mean = output_mean.unsqueeze(1)
        output_mean = output_mean.repeat(1, 3, 1)
        diff_from_mean = color_seqs - output_mean
        scores = torch.matmul(diff_from_mean, covar_matrix)
        diff_transpose = diff_from_mean.transpose(1, 2)
        scores = torch.matmul(scores, diff_transpose) # dim=(batch_size, 3, color_dim) where 3 is the nber of colors
        #scores = -torch.diag(scores)
        # we just need the diagonals for each matrix per batch
        # to do check if shuld be minus
        scores = -torch.diagonal(scores, dim1=-1, dim2=-2)  # dim=(batch_size, 3)
        #distribution = self.logsoftmax(scores)
        #return distribution
        return scores

    def init_hidden_and_context(self):
        # first 2 because one vector for each direction.
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))


class LiteralListener(ColorAgent):

    def __init__(self, vocab, color_dim=54, **base_kwargs):
        """
        Neural listener. Consumes captions and returns a probability distribution for the three
        colors of being the target color.
        """
        super().__init__(vocab, **base_kwargs)
        self.color_dim = color_dim
        self.model = None
        self.loss = nn.CrossEntropyLoss()

    def build_graph(self):
        self.model = CaptionEncoder(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, vocab_size=self.vocab_size,
                                    color_dim=self.color_dim, embedding=self.embedding)
        return self.model

    def predict(self, color_seqs, word_seqs, device=None):
        """
        Produces and tracks model outputs
        :param color_seqs:
        :param word_seqs:
        :param device:
        :return:
        tensor(nber items, 3). Each row gives the proba given to colors 1, 2 and 3 given the corresponding utterance in
        in word_seqs.
        """
        # todo: check if need to be kept, or modified in a more vectorized fashion
        device = self.device if device is None else torch.device(device)
        dataset = self.build_dataset(color_seqs, word_seqs)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        self.model.to(device)
        self.model.eval()
        model_outputs = torch.tensor([]).to(device)
        with torch.no_grad():

            # for i, feature in enumerate(zip(word_seqs, color_seqs)):
            #     caption, colors = feature
            #     caption = torch.tensor([caption], dtype=torch.long)
            #     colors = torch.tensor([colors], dtype=torch.float)
            #     model_output_np = self.evaluate_iter((caption, colors)).view(-1).numpy()
            #     model_outputs.append(model_output_np)
            for batch_colors, batch_words, _, _ in dataloader:
                batch_colors = batch_colors.to(device)
                batch_words = batch_words.to(device)
                output = self.model(color_seqs=batch_colors, word_seqs=batch_words)
                model_outputs = torch.cat((model_outputs, output), 0)

        softmax = nn.Softmax(dim=1)
        probabilities = softmax(model_outputs)
        self.model.train()
        return probabilities

    def train_iter(self, caption_tensor, color_tensor, target, criterion):
        """
        Iterates through a single training pair, querying the model, getting a loss and
        updating the parameters. (TODO: add some kind of batching to this).
        """
        # todo: check if need to be kept
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
        # todo: check if need to be kept
        with torch.no_grad():
            caption_tensor, color_tensor = pair
            model_output = self.model(color_tensor, caption_tensor)

            model_output = model_output.view(1, -1)
            return model_output

    def evaluate(self, color_seqs, word_seqs, device=None):
        """
        For evaluation of the listener, the accuracy is calculated. There is no need to include the
        colors used for producing the description as, by construction, the target color is always at the third place.
        The listener needs to select the third color to be accurate.
        :param color_seqs:list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.
        :param word_seqs: list of lists of utterances. A tokenized list of words.
        :param device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.
        :return:
        dict, {"accuracy": float}
        """
        probabilities = self.predict(color_seqs, word_seqs, device=device)
        color_predicted = torch.argmax(probabilities, dim=1)
        correct_predictions = (color_predicted == 2).sum().item()
        accuracy = correct_predictions / color_predicted.shape[0]
        return {'accuracy': accuracy}

    def score(self, color_seqs, word_seqs, device=None):
        """
        Alias for `listener_accuracy`. This method is included to
        make it easier to use sklearn cross-validators, which expect
        a method called `score`.

        """
        accuracy = self.evaluate(color_seqs, word_seqs, device=device)['accuracy']
        return accuracy

    def _get_loss_from_batch(self, batch):
        """
        We overwrite this method as we're dealing with the listener here.
        :param batch:
        """
        # todo: check code here for dealing with batch
        #  y_batch = batch[: -1]
        y_batch =batch[0]
        X_batch = batch[-1]
        batch_preds = self.model(color_seqs=y_batch, word_seqs=X_batch)

        # The expected distribution should be [0, 0, 1] giving a 1 probability to the last color, which is by
        # construction the target color
        expected_distribution = torch.ones(self.batch_size, dtype=int) * 2
        #todo: batch_pred(m, voc, nwords) and ybatch (m, nwords)
        try:
            err = self.loss(batch_preds, expected_distribution)
        except ValueError:
            # last iteration of the batch has smaller dimension than self.batch_size
            current_size = X_batch.shape[0]
            expected_distribution = torch.ones(current_size, dtype=int) * 2
            err = self.loss(batch_preds, expected_distribution)
        return err

