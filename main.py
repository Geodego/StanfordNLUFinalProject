from project.object.task_handler import TaskHandler
from project.data.study.database import ColorDB


if __name__ == '__main__':
    task = TaskHandler()
    # output = task.initialize_optimal_agent(hyper_id=3, action='hyper')
    # agent, colors_dev, seqs_dev = (output[k] for k in ['model', 'colors_dev', 'seqs_dev'])
    # d = agent.beam_search(colors_dev)

    #corpus_word_count = None,
    # max_iter = None, split_rate = None, prev_split = False, eta = 0.001, batch_size = 1024,
    # glove_dim = 50, n_attention = 1, feed_forward_size = 75, optimizer = 'Adam',
    # early_stopping = False
    #param_fixed = {'early_stopping': True, 'optimizer': 'Adam', 'glove_dim': 100, 'batch_size': 32}
    #param_grid = {'eta': [0.001, 0.005, 0.01, 0.015], 'hidden_dim': [50, 100, 150]}

    #print('done, processes all done')
    # next round
    #'glove_dim': [None, 100], 'batch_size': [16, 32, 64, 128]
    # next round
    #'early_stopping': [True, False]

    # Transfo: Hidden size: H. n attentions A=H/64. feedforward size F=4H. Number of layers L.
    # 1st round: with glove
    param_fixed = {'early_stopping': True, 'optimizer': 'Adam', 'glove_dim': 100, 'batch_size': 256,
                   'hidden_dim': 100, 'n_attention': 1, 'num_layers': 1}
    param_grid = {'eta': [0.001, 0.005, 0.01, 0.015], 'feedforward_size': [75, 200, 400, 600]}

    output = task.train_and_save_agent(hyper_id=3, training_data_id=3, save_memory=True)
    model, colors_dev, seqs_dev = (output[k] for k in ['model', 'colors_dev', 'seqs_dev'])
    score = model.evaluate(colors_dev, seqs_dev)
    pass
    # 2nd round: no glove
    param_fixed = {'early_stopping': True, 'optimizer': 'Adam', 'glove_dim': None, 'batch_size': 256,
                  'num_layers': 1, 'hidden_dim': 128, 'eta': 0.001}
    param_grid = {'n_attention': [1, 2], 'feed_forward_size': [512, 600, 800]}

    # 3 round batch/size early stopping
    param_fixed = {'optimizer': 'Adam', 'batch_size': 128, 'early_stopping': True,
                   'num_layers': 2, 'hidden_dim': 100, 'eta': 0.001, 'n_attention': 2, 'feed_forward_size': 400}
    param_grid = {'glove_dim': [100, None]}
    c = task.hyperparameters_search(3, param_fixed, param_grid, save_results=True)
    # 4th round Adadelta
    param_fixed = {'optimizer': 'Adadelta', 'glove_dim': 100, 'early_stopping': True, 'batch_size': 256,
                   'num_layers': 2, 'hidden_dim': 100, 'n_attention': 2, 'feed_forward_size': 400}
    param_grid = {'eta': [0.10, 0.2, 0.3]}
    #5th round refine glove/no glove, ffw size
    # 6th round increase the number of layers
    param_fixed = {'optimizer': 'Adam', 'early_stopping': True, 'glove_dim': None, 'eta': 0.001, 'batch_size': 128,
                   'hidden_dim': 128, 'n_attention': 2, 'num_layers': 5, 'feed_forward_size': 512}
    param_grid = {'eta': [0.0005, 0.001, 0.0015]}
    # 7th round increase 'hidden-dim'
    param_fixed = {'optimizer': 'Adam', 'batch_size': 128, 'early_stopping': True, 'glove_dim': None,
                   'num_layers': 2, 'eta': 0.001, 'feed_forward_size': 768}
    param_grid = {'hidden_dim': [192], 'n_attention': [3]}
