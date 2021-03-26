from .torch_color_describer import Encoder, EncoderDecoder, ContextualColorDescriber
from ..modules.textual_heads import TransformerDecoder
from ..utils.beam_search import AutoRegressiveBeamSearch
import torch
import functools


class TransformerDescriber(ContextualColorDescriber):
    """
    Base color describer, encoder initializes embedding, doesn't handle the colorized feature of HW4 yet
    embeddings: pretrained embeddings
    """

    def __init__(self, *args, num_layers=1, n_attention=None, feedforward_size=None,
                 max_caption_length=100, max_decoding_steps=20, **kwargs):
        """

        :param args:
        :param num_layers:
        :param n_attention:
        :param feedforward_size:
        :param kwargs:
        """
        self.num_layers = num_layers
        self.feedforward_size = feedforward_size
        self.n_attention = n_attention
        self.max_caption_length = max_caption_length
        super(TransformerDescriber, self).__init__(*args, **kwargs)
        self.beam_search = AutoRegressiveBeamSearch(
            self.end_index, beam_size=2, max_steps=max_decoding_steps
        )

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
                                     max_caption_length=self.max_caption_length, device=self.device, **kwargs)

        encoder.to(self.device)
        decoder.to(self.device)  # makes sure the decoder is where we want
        self.embed_dim = decoder.embedding.words.embedding_dim
        # Return a `TransformerEncoderDecoder` that uses Encoder and TransformerTextualHead
        output = TransformerEncoderDecoder(encoder, decoder)
        self.model = output
        return output

    def predict(self, color_seqs, max_length=20, device=None):
        """
        Predict new sequences based on the color contexts in
        `color_seqs`. The dynamic is modified from the one used for RNN where only the previous word and the previous
        hidden state are used as input (the previous hidden state has the information of the sentence creation up to
        now). With the Transformer decoder, the sentence up to now needs to be used as input.

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
            try:
                hidden = self.model.encoder(color_seqs)
            except AttributeError:
                # we are in RSASpeakerListener model
                hidden = self.model.encoder_decoder.encoder(color_seqs)

            # Start with START_SYMBOL for all examples:
            decoder_input = [[self.start_index]] * len(color_seqs)
            decoder_input = torch.LongTensor(decoder_input)
            decoder_input = decoder_input.to(device)

            preds.append(decoder_input)

            # Now move through the remaiming timesteps using the
            # previous timestep to predict the next one:
            for i in range(1, max_length):
                output = self.model(
                    color_seqs=color_seqs,
                    word_seqs=decoder_input,
                    seq_lengths=None,
                    hidden=hidden)

                # Always take the highest probability token to
                # be the prediction:
                p = output.argmax(2)
                # get the last word predicted
                last_word = p[:, -1].reshape(-1, 1)
                preds.append(last_word)
                decoder_input = torch.cat((decoder_input, last_word), 1)

        # Convert all the predictions from indices to elements of
        # `self.vocab`:
        preds = torch.cat(preds, axis=1)
        preds = [self._convert_predictions(p) for p in preds]

        self.model.to(self.device)

        return preds

    def beam_search_prev(self, color_seqs, max_length=20, beam_size=2, device=None):
        """
        Predict new sequences based on the color contexts in
        `color_seqs` using beam search.
        :param color_seqs:
        :param max_length:
        :param device:
        :param beam_size:
        :return:
        """
        # todo: remove?
        device = self.device if device is None else torch.device(device)

        color_seqs = torch.FloatTensor(color_seqs)
        color_seqs = color_seqs.to(device)

        self.model.to(device)

        self.model.eval()

        beam_pred = []  # store the captions for beam search
        beam_proba = [[1] for i in range(beam_size)]   # store the probabilities of captions in beam_pred

        with torch.no_grad():
            # Get the hidden representations from the color contexts:
            try:
                hidden = self.model.encoder(color_seqs)
            except AttributeError:
                # we are in RSASpeakerListener model
                hidden = self.model.encoder_decoder.encoder(color_seqs)

            # Start with START_SYMBOL for all examples:
            decoder_input = [[self.start_index]] * len(color_seqs)
            decoder_input = [decoder_input] * beam_size  # add one dimension so that we can handle beam search
            decoder_input = torch.LongTensor(decoder_input)
            decoder_input = decoder_input.to(device)

            beam_pred.append(decoder_input)

            # Now move through the remaiming timesteps using the
            # previous timestep to predict the next one:
            for i in range(1, max_length):
                output = self.model(
                    color_seqs=color_seqs,
                    word_seqs=decoder_input,
                    seq_lengths=None,
                    hidden=hidden)

                # Always take the  beam_size highest probability token to
                # be the prediction:
                p = torch.topk(output, k=beam_size)
                # get the last word predicted
                last_word = p[:, -1].reshape(-1, 1)
                beam_pred.append(last_word)
                #todo: finish beam search here
                decoder_input = torch.cat((decoder_input, last_word), 1)

        # Convert all the predictions from indices to elements of
        # `self.vocab`:
        preds = torch.cat(preds, axis=1)
        preds = [self._convert_predictions(p) for p in preds]

        self.model.to(self.device)

        return preds

    def predict_beam_search(self, color_seqs, device=None):
        device = self.device if device is None else torch.device(device)

        color_seqs = torch.FloatTensor(color_seqs)
        color_seqs = color_seqs.to(device)

        self.model.to(device)

        self.model.eval()
        batch_size = color_seqs.size(0)
        start_predictions = color_seqs.new_full((batch_size,), self.start_index).long()
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        beam_search_step = functools.partial(self.beam_search_step, color_seqs)
        all_top_k_predictions, _ = self.beam_search.search(
            start_predictions, beam_search_step
        )
        best_beam = all_top_k_predictions[:, 0, :]
        output_dict = {"predictions": best_beam}
        return output_dict

    def beam_search_step(
        self, visual_features: torch.Tensor, partial_captions: torch.Tensor
    ) -> torch.Tensor:
        #todo: keep?
        r"""
        Given visual features and a batch of (assumed) partial captions, predict
        the distribution over vocabulary tokens for next time-step. This method
        is used by :class:`~virtex.utils.beam_search.AutoRegressiveBeamSearch`.

        Parameters
        ----------
        projected_visual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., textual_feature_size)``
            with visual features already projected to ``textual_feature_size``.
        partial_captions: torch.Tensor
            A tensor of shape ``(batch_size * beam_size, timesteps)``
            containing tokens predicted so far -- one for each beam. We need all
            prior predictions because our model is auto-regressive.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size * beam_size, vocab_size)`` -- output
            distribution over tokens for next time-step.
        """

        # Expand and repeat image features while doing beam search.
        batch_size, channels, height = visual_features.size()  # todo: check, remove width
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            # shape: (batch_size * beam_size, channels, height)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, channels, height)

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a time-step. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        output_logits = self.model(visual_features, partial_captions, caption_lengths)
        # Keep features for last time-step only, we only care about those.
        output_logits = output_logits[:, -1, :]

        # Return logprobs as required by `AutoRegressiveBeamSearch`.
        # shape: (batch_size * beam_size, vocab_size)
        next_logprobs = torch.nn.functional.log_softmax(output_logits, dim=1)

        # Set logprobs of last predicted tokens as high negative value to avoid
        # repetition in caption.
        for index in range(batch_size * beam_size):
            next_logprobs[index, partial_captions[index, -1]] = -10000

        return next_logprobs


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
            seq_lengths = torch.ones(word_seqs.shape[0]) * word_seqs.shape[1]
        output_logits = self.decoder(visual_features=hidden, caption_tokens=word_seqs, caption_lengths=seq_lengths)
        # output_logits.shape: [batch_size, max_num_word_in_batch, vocab size]

        if self.training:
            # Drop the final element:
            output = output_logits[:, : -1, :]
            # Reshape for the sake of the loss function:
            output_caption = output.transpose(1, 2)
            return output_caption
        else:
            return output_logits
