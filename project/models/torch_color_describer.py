import itertools

import nltk.translate.bleu_score
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd

from .agents_base import ColorSpeaker
from ..utils import utils
from ..modules.embedding import WordEmbedding


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Fall 2020"


class Encoder(nn.Module):
    def __init__(self, color_dim, hidden_dim):
        """
        Simple Encoder model based on a GRU cell.

        Parameters
        ----------
        color_dim : int

        hidden_dim : int

        """
        super().__init__()
        self.color_dim = color_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(
            input_size=self.color_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)

    def forward(self, color_seqs):
        """
        Parameters
        ----------
        color_seqs : torch.FloatTensor
            The shape is `(m, n, p)` where `m` is the batch_size,
             `n` is the number of colors in each context, and `p` is
             the color dimensionality.

        Returns
        -------
        hidden : torch.FloatTensor
            These are the final hidden state of the RNN for this batch,
            shape `(m, p) where `m` is the batch_size and `p` is
             the color dimensionality.

        """
        output, hidden = self.rnn(color_seqs)
        return hidden


class Decoder(WordEmbedding):

    def __init__(self, hidden_dim, *args, **kwargs):
        """
        Simple Decoder model based on a GRU cell. The hidden
        representations of the GRU are passed through a dense linear
        layer, and those logits are used to train the language model
        according to a softmax objective in `ContextualColorDescriber`.

        Parameters
        ----------

        hidden_dim : int

        """
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, word_seqs, seq_lengths=None, hidden=None, target_colors=None):
        """
        Core computation for the model.

        Parameters
        ----------
        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch. The `forward`
            method uses `self.get_embeddings` to mape these indices to their
            embeddings.

        seq_lengths : torch.LongTensor
            Shape (m, ) where `m` is the number of examples in the batch.

        hidden : torch.FloatTensor
            Shape `(m, self.hidden_dim)`. When training, this is always the
            final state of the `Encoder`. During prediction, this might be
            recursively computed as the sequence is processed.

        target_colors : torch.FloatTensor
            Dimension (m, c), where m is the number of examples and
            c is the dimensionality of the color representations.

        Returns
        -------
        output : torch.FloatTensor
            The full sequence of outputs states. When we are training, the
            shape is `(m, hidden_dim, k)` to accommodate the expectations
            of the loss function. During prediction, the shape is
            `(m, k, hidden_dim)`. In both cases, m is the number of examples in
            the batch and `k` is the maximum length of sequences in this batch.

        hidden : torch.FloatTensor
            The final output state of the network. Shape `(m, hidden_dim)`
            where m is the number of examples in the batch.

        """
        embs = self.get_embeddings(word_seqs, target_colors=target_colors)

        if self.training:
            # Packed sequence for performance:
            embs = torch.nn.utils.rnn.pack_padded_sequence(
                embs,
                batch_first=True,
                lengths=seq_lengths,
                enforce_sorted=False)
            # RNN forward:
            output, hidden = self.rnn(embs, hidden)
            # Unpack:
            output, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)
            # Output dense layer to get logits:
            output = self.output_layer(output)
            # Drop the final element:
            output = output[:, : -1, :]
            # Reshape for the sake of the loss function:
            output = output.transpose(1, 2)
            return output, hidden
        else:
            output, hidden = self.rnn(embs, hidden)
            output = self.output_layer(output)
            return output, hidden


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        This class knits the `Encoder` and `Decoder` into a single class
        that serves as the model for `ContextualColorDescriber`. This is
        largely a convenience: it means that `ContextualColorDescriber`
        can use a single `model` argument, and it allows us to localize
        the core computations in the `forward` method of this class.

        Parameters
        ----------
        encoder : `Encoder`

        decoder : `Decoder`

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, color_seqs, word_seqs, seq_lengths, hidden=None):
        """This is the core method for this module. It has a lot of
        arguments mainly to make it easy to create subclasses of this
        class that do interesting things without requring modifications
        to the `fit` method of `ContextualColorDescriber`.

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

        hidden : torch.FloatTensor
            Dimension (m, h) where m is the number of examples and h is
            the dimensionality of the hidden representations of the model.
            This value is returned only when the model is in eval mode.

        """
        if hidden is None:
            hidden = self.encoder(color_seqs)
        output, hidden = self.decoder(
            word_seqs, seq_lengths=seq_lengths, hidden=hidden)
        if self.training:
            return output
        else:
            return output, hidden


class ContextualColorDescriber(ColorSpeaker):

    def __init__(self, vocab, **base_kwargs):
        """
        The primary interface to modeling contextual colors speakers.

        Parameters
        ----------
        vocab : list of str
            This should be the vocabulary. It needs to be aligned with
            `embedding` in the sense that the ith element of vocab
            should be represented by the ith row of `embedding`.
        **base_kwargs
            For details, see `ColorAgent`.
        """
        super(ContextualColorDescriber, self).__init__(vocab, **base_kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.model = None
        self.build_graph()

    def build_graph(self):
        """
        The core computation graph. This method is called by `fit` to set
        the `self.model` attribute.

        Returns
        -------
        `EncoderDecoder` built from `Encoder` and `Decoder`

        """
        encoder = Encoder(
            color_dim=self.color_dim,
            hidden_dim=self.hidden_dim)

        decoder = Decoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            freeze_embedding=self.freeze_embedding)

        self.embed_dim = decoder.embed_dim
        self.model = EncoderDecoder(encoder, decoder)

        return self.model

    def predict(self, color_seqs, max_length=20, device=None):
        """
        Predict new sequences based on the color contexts in
        `color_seqs`.

        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        max_length : int
            Length of the longest sequences to create.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list of str

        """
        device = self.device if device is None else torch.device(device)

        color_seqs = torch.FloatTensor(color_seqs)
        color_seqs = color_seqs.to(device)

        self.model.to(device)

        self.model.eval()

        preds = []

        with torch.no_grad():
            # Get the hidden representations from the color contexts:
            hidden = self.model.encoder(color_seqs)

            # Start with START_SYMBOL for all examples:
            decoder_input = [[self.start_index]] * len(color_seqs)
            decoder_input = torch.LongTensor(decoder_input)
            decoder_input = decoder_input.to(device)

            preds.append(decoder_input)

            # Now move through the remaiming timesteps using the
            # previous timestep to predict the next one:
            for i in range(1, max_length):
                output, hidden = self.model(
                    color_seqs=color_seqs,
                    word_seqs=decoder_input,
                    seq_lengths=None,
                    hidden=hidden)

                # Always take the highest probability token to
                # be the prediction:
                p = output.argmax(2)
                preds.append(p)
                decoder_input = p

        # Convert all the predictions from indices to elements of
        # `self.vocab`:
        preds = torch.cat(preds, axis=1)
        preds = [self._convert_predictions(p) for p in preds]

        self.model.to(self.device)

        return preds

    def _convert_predictions(self, pred):
        rep = []
        for i in pred:
            i = i.item()
            rep.append(self.index2word[i])
            if i == self.end_index:
                return rep
        return rep

    def predict_proba(self, color_seqs, word_seqs, device=None):
        """
        Calculate the predicted probabilties of the sequences in
        `word_seqs` given the color contexts in `color_seqs`.

        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of list of int
            Dimension m, the number of examples. The length of each
            sequence can vary.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.


        Returns
        -------
        list of lists of predicted probabilities. In other words,
        for each example, at each timestep, there is a probability
        distribution over the entire vocabulary.

        """
        device = self.device if device is None else torch.device(device)

        dataset = self.build_dataset(color_seqs, word_seqs)

        dataloader = self._build_dataloader(dataset, shuffle=False)

        self.model.to(device)

        self.model.eval()

        softmax = nn.Softmax(dim=2)

        start_probs = np.zeros(self.vocab_size)
        start_probs[self.start_index] = 1.0

        all_probs = []

        with torch.no_grad():

            for batch_colors, batch_words, batch_lens, targets in dataloader:

                batch_colors = batch_colors.to(device)
                batch_words = batch_words.to(device)
                batch_lens = batch_lens.to(device)

                # need to handle case of RNN based models returning an output and a hidden state and transformer based
                # models returning just an output.
                try:
                    output, _ = self.model(
                        color_seqs=batch_colors,
                        word_seqs=batch_words,
                        seq_lengths=batch_lens)
                except ValueError:
                    output = self.model(
                        color_seqs=batch_colors,
                        word_seqs=batch_words,
                        seq_lengths=batch_lens)

                probs = softmax(output)
                probs = probs.cpu().numpy()
                probs = np.insert(probs, 0, start_probs, axis=1)
                all_probs += [p[: n] for p, n in zip(probs, batch_lens)]

        self.model.to(self.device)

        return all_probs

    def perplexities(self, color_seqs, word_seqs, device=None):
        """
        Compute the perplexity of each sequence in `word_seqs`
        given `color_seqs`. For a sequence of conditional probabilities
        p1, p2, ..., pN, the perplexity is calculated as

        (p1 * p2 * ... * pN)**(-1/N)

        Parameters
        ----------
        color_seqs : list of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of list of int
            Dimension m, the number of examples, and the length of
            each sequence can vary.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list of float

        """
        probs = self.predict_proba(color_seqs, word_seqs, device=device)
        scores = []
        for pred, seq in zip(probs, word_seqs):
            # Get the probabilities corresponding to the path `seq`:
            s = np.array([t[self.word2index.get(w, self.unk_index)]
                          for t, w in zip(pred, seq)])
            scores.append(s)
        perp = [np.prod(s) ** (-1 / len(s)) for s in scores]
        return perp

    def listener_predict_one(self, context, seq, device=None):
        """
        The listener choose the color corresponding to the minimum perplexity. Given all permutations of
        the three colors, the color chosen isthe last of the 3 color in the color context that returned the minimum
        perplexity. This is coherent with our model training, for which the target color was always put as the last of
        the 3 colors.
        """
        context = np.array(context)
        n_colors = len(context)

        # Get all possible context orders:
        indices = list(range(n_colors))
        orders = [list(x) for x in itertools.permutations(indices)]

        # All contexts as color sequences:
        contexts = [context[x] for x in orders]

        # Repeat the single utterance the needed number of times:
        seqs = [seq] * len(contexts)

        # All perplexities:
        perps = self.perplexities(contexts, seqs, device=device)

        # Ranking, using `order_indices` rather than colors and
        # index sequences to avoid sorting errors from some versions
        # of Python:
        order_indices = range(len(orders))
        ranking = sorted(zip(perps, order_indices))

        # Return the minimum perplexity, the chosen color, and the
        # index of the chosen color in the original context:
        min_perp, order_index = ranking[0]
        pred_color = contexts[order_index][-1]
        pred_index = orders[order_index][-1]
        return min_perp, pred_color, pred_index

    def listener_accuracy(self, color_seqs, word_seqs, device=None):
        """
        Compute the "listener accuracy" of the model for each example.
        For the ith example, this is defined as

        prediction = max_{c in C_i} P(word_seq[i] | c)

        where C_i is every possible permutation of the three colors in
        color_seqs[i]. We take the model's prediction to be correct
        if it chooses a c in which the target is in the privileged final
        position in the color sequence. (There are two such c's, since
        the distractors can be in two orders; we give full credit if one
        of these two c's is chosen.)

        Parameters
        ----------
        color_seqs : list of lists of list of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of list of int
            Dimension m, the number of examples, and the length of
            each sequence can vary.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list of float

        """
        correct = 0
        for color_seq, word_seq in zip(color_seqs, word_seqs):
            target_index = len(color_seq) - 1
            min_perp, pred, pred_index = self.listener_predict_one(color_seq, word_seq, device=device)
            correct += int(target_index == pred_index)
        return correct / len(color_seqs)

    def get_all_results(self, color_seqs, word_seqs, device=None):
        """
        Returns a list of results for all examples with 1 if the listener successfully finds the target color and 0
        otherwise.
        """
        results = []
        for color_seq, word_seq in zip(color_seqs, word_seqs):
            target_index = len(color_seq) - 1
            min_perp, pred, pred_index = self.listener_predict_one(color_seq, word_seq, device=device)
            results.append(int(target_index == pred_index))
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
        return self.listener_accuracy(color_seqs, word_seqs, device=device)

    def corpus_bleu(self, color_seqs, word_seqs):
        """
        Calculate the corpus BLEU score achieved by `model` with respect
        to `color_seqs` and `word_seqs`, using just unigrams.


        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of lists of utterances
            A tokenized list of words.

        Returns
        -------
        float

        """
        # Ideally, we would have multiple references for each context,
        # but alas we have only one:
        refs = [[seq] for seq in word_seqs]

        # Predict some utterances:
        preds = self.predict(color_seqs)

        # Calculate a unigrams-only BLEU score:
        bleu = nltk.translate.bleu_score.corpus_bleu(
            refs, preds, weights=(1,))

        return bleu

    def evaluate(self, color_seqs, word_seqs, device=None):
        """
        Full evaluation for the bake-off. Uses `listener_accuracy`
        and colors_corpus_bleu`.

        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.

        word_seqs : list of lists of utterances. A tokenized list of words.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        dict, {"listener_accuracy": float, 'corpus_bleu': float}

        """
        acc = self.listener_accuracy(color_seqs, word_seqs, device=device)
        bleu = self.corpus_bleu(color_seqs, word_seqs)
        return {"listener_accuracy": acc, 'corpus_bleu': bleu}


def create_example_dataset(group_size=100, vec_dim=2):
    """
    Creates simple datasets in which the inputs are three-vector
    sequences and the outputs are simple character sequences, with
    the range of values in the final vector in the input determining
    the output sequence. For example, a single input/output pair
    will look like this:

    [[0.44, 0.51], [0.87, 0.89], [0.1, 0.2]],  ['<s>', 'A', '</s>']

    The sequences are meaningless, as are their lengths (which were
    chosen only to be different from each other).

    """
    import random

    groups = ((0.0, 0.2), (0.4, 0.6), (0.8, 1.0))
    vocab = ['<s>', '</s>', 'A', 'B', '$UNK']
    seqs = [
        ['<s>', 'A', '</s>'],
        ['<s>', 'A', 'B', '</s>'],
        ['<s>', 'B', 'A', 'B', 'A', '</s>']]

    color_seqs = []
    word_seqs = []
    for i, ((l, u), seq) in enumerate(zip(groups, seqs)):

        dis_indices = list(range(len(groups)))
        dis_indices.remove(i)
        random.shuffle(dis_indices)
        disl1, disu1 = groups[dis_indices[0]]
        disl2, disu2 = groups[dis_indices[1]]

        for _ in range(group_size):
            target = utils.randvec(vec_dim, l, u)
            dis1 = utils.randvec(vec_dim, disl1, disu1)
            dis2 = utils.randvec(vec_dim, disl2, disu2)
            context = [dis1, dis2, target]
            color_seqs.append(context)

        word_seqs += [seq for _ in range(group_size)]

    return color_seqs, word_seqs, vocab


def simple_example(group_size=100, vec_dim=2):
    from sklearn.model_selection import train_test_split

    utils.fix_random_seeds()

    color_seqs, word_seqs, vocab = create_example_dataset(
        group_size=group_size, vec_dim=vec_dim)

    X_train, X_test, y_train, y_test = train_test_split(
        color_seqs, word_seqs)

    mod = ContextualColorDescriber(vocab)

    print(mod)

    mod.fit(X_train, y_train)

    preds = mod.predict(X_test)

    mod.predict_proba(X_test, y_test)

    correct = 0
    for y, p in zip(y_test, preds):
        correct += int(y == p)

    print("\nExact sequence: {} of {} correct".format(correct, len(y_test)))

    lis_acc = mod.listener_accuracy(X_test, y_test)

    print("\nListener accuracy {}".format(lis_acc))

    return lis_acc


