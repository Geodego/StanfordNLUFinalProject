from project.object.task_handler import TaskHandler
from project.data.study.database import ColorDB


if __name__ == '__main__':
    task = TaskHandler()
    #corpus_word_count = None,
    # max_iter = None, split_rate = None, prev_split = False, eta = 0.001, batch_size = 1024,
    # glove_dim = 50, n_attention = 1, feed_forward_size = 75, optimizer = 'Adam',
    # early_stopping = False
    param_fixed = {'early_stopping': True, 'optimizer': 'Adam', 'glove_dim': 100, 'batch_size': 32}
    param_grid = {'eta': [0.001, 0.005, 0.01, 0.015], 'hidden_dim': [50, 100, 150]}
    c = task.hyperparameters_search(1, param_fixed, param_grid, save_results=True)
    print('done, processes all done')
    # next round
    #'glove_dim': [None, 100], 'batch_size': [16, 32, 64, 128]
    # next round
    #'early_stopping': [True, False]
