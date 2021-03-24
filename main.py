from project.object.task_handler import TaskHandler
from project.data.study.database import ColorDB


if __name__ == '__main__':
    task = TaskHandler()
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

    pass
    # 2nd round: no glove
    param_fixed = {'early_stopping': True, 'optimizer': 'Adam', 'glove_dim': None, 'batch_size': 256,
                  'num_layers': 1, 'hidden_dim': 128, 'eta': 0.001}
    param_grid = {'n_attention': [1, 2], 'feedforward_size': [512, 600, 800]}
    c = task.hyperparameters_search(3, param_fixed, param_grid, save_results=True)
    # 3 round batch/size early stopping
    # 4th round Adadelta
