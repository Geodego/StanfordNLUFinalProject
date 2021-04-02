import torch
import numpy as np
import pandas as pd
from torch import nn
from .agents_base import ColorListener
from ..modules.embedding import WordEmbedding


class CaptionEncoder(WordEmbedding):
    """
    Implements the Monroe et al., 2017 color caption encoder. The CaptionEncoder takes a list of possible colors and a
    caption and converts the caption into a distribution over the colors.
    """

    def __init__(self, hidden_dim, color_dim, device=None, *args, **kwargs):
        """

        :param hidden_dim: dimension used for the hidden state of the LSTM.
        :param color_dim: number of dimensions of the input color vectors, the mean
                vector and the covariance matrix. Usually will be 54.
        :param device:
        :param args:
        :param kwargs:
        """
        super(CaptionEncoder, self).__init__(*args, **kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(self.embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.mean = nn.Linear(2 * hidden_dim, color_dim)

        covar_dim = color_dim * color_dim
        self.covariance = nn.Linear(2 * hidden_dim, covar_dim)
        self.covariance.bias.data = torch.tensor(np.eye(color_dim), dtype=torch.float).flatten().to(device)

        self.color_dim = color_dim
        self.hidden_dim = hidden_dim

    def forward(self, color_seqs, word_seqs, seq_lengths, is_embedded=False):
        """
        Turns caption into distribution over colors.
        :param color_seqs: torch.FloatTensor
            Dimension (m, 3, c), where m is the number of examples and
            c is the dimensionality of the color representations.
        :param word_seqs: torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch. The `forward`
            method uses `self.get_embeddings` to mape these indices to their
            embeddings.
        :param seq_lengths: torch.LongTensor
            Shape (m, ) where `m` is the number of examples in the batch.
        :param is_embedded: if true no need to use self.get_embeddings(word_seqs). Mainly for CaptionEncoder
        is used as part of a more complex model.
        :return:
        """
        batch_size = color_seqs.shape[0]
        if not is_embedded:
            embeddings = self.get_embeddings(word_seqs)
        else:
            embeddings = word_seqs

        embeddings.to(self.device)
        # pack sequence for performance
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, batch_first=True, lengths=seq_lengths,
                                                       enforce_sorted=False)
        # rnn forward
        output, _ = self.lstm(embeddings)

        # unpack
        output, seq_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output.to(self.device)

        # we only care about last output (first dim is batch size)
        # here we are concatenating the the last output vector of the forward direction (at index -1)
        # and the last output vector of the first direction (at index 0)
        output = torch.cat((output[:, -1, :self.hidden_dim],
                            output[:, 0, self.hidden_dim:]), 1)

        output_mean = self.mean(output)
        output_covariance = self.covariance(output)
        covar_matrix = output_covariance.reshape(batch_size, -1, self.color_dim)  # make it a square matrix again

        # now compute score: -(f-mu)^T Sigma (f-mu)
        output_mean = output_mean.unsqueeze(1)
        output_mean = output_mean.repeat(1, 3, 1)
        diff_from_mean = color_seqs - output_mean
        scores = torch.matmul(diff_from_mean, covar_matrix)
        diff_transpose = diff_from_mean.transpose(1, 2)
        scores = torch.matmul(scores, diff_transpose)  # dim=(batch_size, 3, color_dim) where 3 is the nber of colors

        # we just need the diagonals for each matrix per batch
        # to do check if shuld be minus
        scores = -torch.diagonal(scores, dim1=-1, dim2=-2)  # dim=(batch_size, 3)
        return scores

    def init_hidden_and_context(self):
        # first 2 because one vector for each direction.
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))


class LiteralListener(ColorListener):

    def __init__(self, vocab, color_dim=54, device=None, **base_kwargs):
        """
        Neural listener. Consumes captions and returns a probability distribution for the three
        colors of being the target color.
        """
        super().__init__(vocab, **base_kwargs)
        self.color_dim = color_dim
        self.model = None
        self.loss = nn.CrossEntropyLoss()
        self.build_graph()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def build_graph(self):
        self.model = CaptionEncoder(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, vocab_size=self.vocab_size,
                                    color_dim=self.color_dim, embedding=self.embedding)
        self.model.to(self.device)
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
        device = self.device if device is None else torch.device(device)
        dataset = self.build_dataset(color_seqs, word_seqs)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        self.model.to(device)
        self.model.eval()
        model_outputs = torch.tensor([]).to(device)
        with torch.no_grad():
            for batch_words, batch_lengths, batch_colors in dataloader:
                batch_colors = batch_colors.to(device)
                batch_words = batch_words.to(device)
                output = self.model(color_seqs=batch_colors, word_seqs=batch_words, seq_lengths=batch_lengths)
                model_outputs = torch.cat((model_outputs, output), 0)

        softmax = nn.Softmax(dim=1)
        probabilities = softmax(model_outputs)
        self.model.train()
        return probabilities

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

    def get_all_results(self, color_seqs, word_seqs, device=None):
        """
        Returns a list of results for all examples with 1 if the listener successfully finds the target color and 0
        otherwise.
        """
        probabilities = self.predict(color_seqs, word_seqs, device=device)
        color_predicted = torch.argmax(probabilities, dim=1)
        results = (color_predicted == 2).numpy().astype('int').tolist()
        return results

    def get_listener_accuracy_per_condition(self, color_seqs, word_seqs, conditions, device=None):
        """
                Returns a DataFame with listener_accuracy per condition

                Parameters
                ----------
                color_seqs : list of lists of list of floats, or np.array
                    Dimension (m, n, p) where m is the number of examples, n is
                    the number of colors in each context, and p is the length
                    of the color representations.

                word_seqs : list of list of int
                    Dimension m, the number of examples, and the length of
                    each sequence can vary.

                conditions: list of condition corresponding to the color sequences in color-seqs (

                device: str or None
                    Allows the user to temporarily change the device used
                    during prediction. This is useful if predictions require a
                    lot of memory and so are better done on the CPU. After
                    prediction is done, the model is returned to `self.device`.

                Returns
                -------
                DataFame with listener_accuracy per condition, columns ['result'] which is the listener accuracy and
                rows ['split', 'close', 'far']

                """
        results = self.get_all_results(color_seqs, word_seqs, device=device)
        df = pd.DataFrame([conditions, results]).T
        df.rename(columns={0: 'condition', 1: 'result'}, inplace=True)
        df = df.astype({'result': 'int32'})
        summary = df.groupby('condition').mean()
        return summary

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
        self.model.to(self.device)
        y_batch = batch[2]
        seq_lengths = batch[1]
        # for utterances we use the last item from batch where the first symbol of sentence initiation has been dropped
        X_batch = batch[0]
        batch_preds = self.model(color_seqs=y_batch, word_seqs=X_batch, seq_lengths=seq_lengths)
        batch_preds.to(self.device)
        # The expected distribution should be [0, 0, 1] giving a 1 probability to the last color, which is by
        # construction the target color
        expected_distribution = torch.ones(self.batch_size, dtype=int) * 2
        expected_distribution = expected_distribution.to(self.device)
        try:
            err = self.loss(batch_preds, expected_distribution)
        except ValueError:
            # last iteration of the batch has smaller dimension than self.batch_size
            current_size = X_batch.shape[0]
            expected_distribution = torch.ones(current_size, dtype=int) * 2
            expected_distribution = expected_distribution.to(self.device)
            err = self.loss(batch_preds, expected_distribution)
        return err
