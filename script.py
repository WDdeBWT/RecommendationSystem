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
    print('--- sin_num: ' + str(sim_num) + ' --- recm_num: ' + str(recm_num) + ' ---')
    print('Recall:     ' + str(eva.recall()))
    print('Precision:  ' + str(eva.precision()))
    print('Coverage:   ' + str(eva.coverage()))
    print('Popularity: ' + str(eva.Popularity()))

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/res50.yaml')
parser.add_argument('-d', '--detail', default='True')
args = parser.parse_args()
show_detail = False if args.detail.lower() == 'false' else True

test(show_detail)