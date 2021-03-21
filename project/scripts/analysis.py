import numpy as np
from project.utils.utils import fix_random_seeds

from project.models.rnn_speaker import ColorizedInputDescriber
from project.models.transformer_based import TransformerDescriber
from project.utils.tools import load_model_states, select_data_with_max_length_sentence
from project.utils.model_tools import load_pretrained_agent


def compare_models(model1_output, model2_output, n_examples=2, sentence_max_length=None):
    """
    Compare model 1 and model2
    :param model1_output: result from get_trained_color_model for model 1
    :param model2_output: result from get_trained_color_model for model 2. If None analyse only model 1
    :param n_examples: number of utterances examples
    :param sentence_max_length: used when we want to restrict sentences used for evaluation to a maximum
    size.
    :return:
    """
    fix_random_seeds()
    output1 = model1_output
    output2 = model2_output
    output_list = [output1] if output2 is None else [output1, output2]
    analysis = dict()  # that's where results of the analysis are saved
    index_predict = np.random.randint(0, 500, n_examples)
    for i, output in enumerate(output_list):
        # score
        model, seqs_dev, colors_dev, seqs_train, colors_train = (
            output[k] for k in ['model', 'seqs_dev', 'colors_dev', 'seqs_train', 'colors_train'])
        if sentence_max_length is not None:
            # we only keep data corresponding to items in seqs_test with length below sentence_max_length
            seqs_dev, colors_dev, seqs_train, colors_train = select_data_with_max_length_sentence(
                sentence_max_length, seqs_dev, colors_dev, seqs_train, colors_train
            )

        model_name = str(model).split('(')[0]
        results = dict()  # that's were we save results for that particular model
        score = model.evaluate(colors_dev, seqs_dev)
        train_score = model.evaluate(colors_train, seqs_train)
        results['train_score'] = train_score
        results['score'] = score

        # check length of predictions
        selected_cols = [colors_dev[i] for i in range(2000)]
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
        print('\t training score: {}'.format(values['train_score']))
        print('\tshortest sentence length: {}'.format(values['min_length']))
        print('\tlonguest sentence length: {}'.format(values['max_length']))

    print('\n' + sep)
    print(' ' * 10 + '*** Utterances examples ***')
    sentences = [k['ex'] for k in analysis.values()]
    if model2_output is not None:
        for ex1, ex2 in zip(sentences[0], sentences[1]):
            print(' '.join(ex1))
            print(' '.join(ex2) + '\n')
    else:
        for ex in sentences[0]:
            print(' '.join(ex))


if __name__ == '__main__':
    speaker1 = ColorizedInputDescriber
    speaker2 = TransformerDescriber
    file_name1 = 'ColorizedInputDescriber_monroe_split'
    file_name2 = 'TransformerDescriber_monroe_split_2'
    # output1 = train_and_save_speaker(speaker1, corpus_word_count=None, file_name=file_name1)
    # output2 = train_and_save_speaker(speaker2, corpus_word_count=None, file_name=file_name2)
    #output1 = get_model_pretrained_output(speaker1, prev_split=True,
    #                                     file_params=file_name1, corpus_word_count=None)
    #output2 = get_model_pretrained_output(speaker2, prev_split=True,
    #                                      file_params=file_name2, corpus_word_count=None,
    #                                      max_caption_length=60)
    #output1 = load_pretrained_agent(agent=speaker1, file_params=file_name1, glove_dim=50)
    output2 = load_pretrained_agent(agent=speaker2, file_params=file_name2, corpus_word_count=None, glove_dim=50)
    compare_models(output2, None)
    exit()
    model = output2['model']
    colors = output2['colors_test']
    model.beam_search(color_seqs=colors)

