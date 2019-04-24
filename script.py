import time

import utils
import model
import evaluator

def test():
    sim_num = 30
    recm_num = 15

    rate_list = utils.read_csv(utils.data_path)[1:]
    train_data, test_data = utils.split_data(rate_list, 1)
    ub_model = model.UserBasedModel(train_data)
    ub_model.get_sim(sim_num)
    recm_table = ub_model.get_recm(recm_num)
    eva = evaluator.Evaluator(rate_list, test_data, recm_table, recm_num)
    print('--- sin_num: ' + str(sim_num) + ' --- recm_num: ' + str(recm_num) + ' ---')
    print('Recall:     ' + str(eva.recall()))
    print('Precision:  ' + str(eva.precision()))
    print('Coverage:   ' + str(eva.coverage()))
    print('Popularity: ' + str(eva.Popularity()))


test()