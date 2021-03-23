from project.object.task_handler import TaskHandler
from project.data.study.database import ColorDB


if __name__ == '__main__':
    task = TaskHandler()
    #corpus_word_count = None,
    # max_iter = None, split_rate = None, prev_split = False, eta = 0.001, batch_size = 1024,
    # glove_dim = 50, n_attention = 1, feed_forward_size = 75, optimizer = 'Adam',
    # early_stopping = False
    param_fixed = {'glove_dim': 100, 'optimizer': 'Adam',
                   'batch_size': 64, 'early_stopping': True}
    param_grid = {'eta': [0.0005, 0.001, 0.005, 0.01], 'hidden_dim': [50, 100, 150]}
    c = task.hyperparameters_search(1, param_fixed, param_grid)
    print('done, processes all done')
