import numpy as np
from project.utils.utils import fix_random_seeds
from project.scripts.pretrain import get_trained_color_speaker, train_and_save_speaker
from project.models.rnn_speaker import ColorizedInputDescriber
from project.models.transformer_based import TransformerDescriber
from project.utils.tools import load_model_states, select_data_with_max_length_sentence


def get_model_pretrained_output(model_class, file_params, corpus_word_count):
    """
    Return the output of get_trained_color_model for model which states were saved in
    file_params
    :param model_class: class of the model which decoder encoder params were saved in file_params
    :param file_params:
    :param corpus_word_count:
    :return:
    """
    fix_random_seeds()
    # go through on iteration to set up the untrained model properly
    output = get_trained_color_speaker(corpus_word_count=corpus_word_count, speaker=model_class,
                                       max_iter=1)
    model = output['model']
    model.encoder_decoder = load_model_states(model.encoder_decoder, file_params)
    output['model'] = model
    return output


def compare_models(model1_output, model2_output, n_examples=2, sentence_max_length=None):
    """
    Compare model 1 and model2
    :param model1_output: result from get_trained_color_model for model 1
    :param model2_output: result from get_trained_color_model for model 2
    :param n_examples: number of utterances examples
    :param sentence_max_length: used when we want to restrict sentences used for evaluation to a maximum
    size.
    :return:
    """
    fix_random_seeds()
    output1 = model1_output
    output2 = model2_output
    analysis = dict()  # that's where results of the analysis are saved
    index_predict = np.random.randint(0, 500, n_examples)
    for i, output in enumerate([output1, output2]):
        # score
        model, seqs_test, cols_test, rawcols_test, texts_test = output.values()
        if sentence_max_length is not None:
            # we only keep data corresponding to items in seqs_test with length below sentence_max_length
            seqs_test, cols_test, rawcols_test, texts_test = select_data_with_max_length_sentence(
                sentence_max_length, seqs_test, cols_test, rawcols_test, texts_test
            )

        model_name = str(model).split('(')[0]
        results = dict()  # that's were we save results for that particular model
        score = model.evaluate(cols_test, seqs_test)
        results['score'] = score

        # check length of predictions
        selected_cols = [cols_test[i] for i in range(2000)]
        predicted = model.predict(selected_cols)

        sentences_length = [len(s) for s in predicted]
        min_sentence = min(sentences_length)
        max_sentence = max(sentences_length)
        results['min_length'] = min_sentence
        results['max_length'] = max_sentence

        # select some predicted utterances
        results['ex'] = [predicted[i] for i in index_predict]

        # focus on longuest sentences
        predicted = [s for s in predicted if len(s) > 8]
        results['ex'] += predicted[:3]


        # save the results in analysis
        analysis[model_name] = results

    sep = '*' * 50
    print('\n' + sep)
    print(' ' * 10 + '*** Scores ***')
    for model, values in analysis.items():
        print('model {}: \n\t score: {}'.format(model, values['score']))
        print('\tshortest sentence length: {}'.format(values['min_length']))
        print('\tlonguest sentence length: {}'.format(values['max_length']))

    print('\n' + sep)
    print(' ' * 10 + '*** Utterances examples ***')
    sentences = [k['ex'] for k in analysis.values()]
    for ex1, ex2 in zip(sentences[0], sentences[1]):
        print(' '.join(ex1))
        print(' '.join(ex2) + '\n')


if __name__ == '__main__':
    speaker1 = ColorizedInputDescriber
    speaker2 = TransformerDescriber
    file_name1 = 'ColorizedInputDescriber_alldata'
    file_name2 = 'TransformerDescriber_alldata_utterhistory'
    #output1 = train_and_save(speaker1, corpus_word_count=None, file_name=file_name1)
    #output2 = train_and_save(speaker2, corpus_word_count=None, file_name=file_name2)
    output1 = get_model_pretrained_output(speaker1, file_params=file_name1, corpus_word_count=None)
    output2 = get_model_pretrained_output(speaker2, file_params=file_name2, corpus_word_count=None)
    compare_models(output1, output2, n_examples=20, sentence_max_length=58)

