from ..utils.model_tools import initialize_agent
from ..scripts.pretrain import train_and_save_agent
import itertools
import logging

logger = logging.getLogger(__name__)


def hyperparameters_search(model_class, param_grid, param_fixed):
    max_accuracy = 0
    best_param = None
    crosses = itertools.product(*list(param_grid.values()))
    total = len(list(crosses))
    for i, values in enumerate(crosses):
        progress = round(i/total * 100)
        logger.info('progress: {}%'.format(progress))
        val_dic = {key: value for key, value in zip(param_grid.keys(), values)}
        all_param = {**val_dic, **param_fixed}
        output = train_and_save_agent(model=model_class, action='hyper', file_name='', silent=True, **all_param)
        if output['accuracy'] > max_accuracy:
            max_accuracy = output['accuracy']
            best_param = val_dic
    print("Best params: {}".format(best_param))
    print("Best accuracy: {}".format(max_accuracy))

    return best_param, max_accuracy




