import time
import argparse

import utils
import model
import evaluator

def test(show_detail):
    sim_num = 30
    recm_num = 15
    data_path = 'sample_data.csv'
    rate_list = utils.read_csv(data_path, show_detail)[1:]
    train_data, test_data = utils.split_data(rate_list, 1, show_detail)
    ub_model = model.UserBasedModel(train_data, show_detail)
    ub_model.get_sim(sim_num)
    recm_table = ub_model.get_recm(recm_num)

    eva = evaluator.Evaluator(rate_list, test_data, recm_table, recm_num)
    recall = eva.recall()
    precision = eva.precision()
    coverage = eva.coverage()
    popularity = eva.popularity()

    log_list = []
    print('--- sim_num: ' + str(sim_num) + ' --- recm_num: ' + str(recm_num) + ' ---')
    log_list.append('--- sim_num: ' + str(sim_num) + ' --- recm_num: ' + str(recm_num) + ' ---')
    print('Recall:     ' + str(recall))
    log_list.append('Recall:     ' + str(recall))
    print('Precision:  ' + str(precision))
    log_list.append('Precision:  ' + str(precision))
    print('Coverage:   ' + str(coverage))
    log_list.append('Coverage:   ' + str(coverage))
    print('Popularity: ' + str(popularity))
    log_list.append('Popularity: ' + str(popularity))
    utils.write_log('log.txt', log_list)


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/res50.yaml')
parser.add_argument('-d', '--detail', default='True')
args = parser.parse_args()
show_detail = False if args.detail.lower() == 'false' else True

test(show_detail)