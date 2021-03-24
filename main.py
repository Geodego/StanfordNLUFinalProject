from project.object.task_handler import TaskHandler
from project.data.study.database import ColorDB


if __name__ == '__main__':
    task = TaskHandler()
    #corpus_word_count = None,
    # max_iter = None, split_rate = None, prev_split = False, eta = 0.001, batch_size = 1024,
    # glove_dim = 50, n_attention = 1, feed_forward_size = 75, optimizer = 'Adam',
    # early_stopping = False
    param_fixed = {'optimizer': 'Adam', 'eta': 0.005, 'batch_size': 32}
    param_grid = {'early_stopping': [True, False], 'hidden_dim': [50, 100], 'glove_dim': [None, 100]}
    c = task.hyperparameters_search(4, param_fixed, param_grid)
    print('done, processes all done')
