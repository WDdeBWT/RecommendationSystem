# fuzzy preference similarity with user based model
import time
import math

import numpy as np

import data
import utils
import evaluator
import model_test


FUZZY_MOD = True
SIM_WEIGHT = 0.8
TEST_SIZE = 1000 # None means full size


def user_based_model(user_id, movie_id, tdata, sim_weight):
    if (user_id not in tdata.user_list) or (movie_id not in tdata.movie_list):
        return 3.5
    ui = tdata.user_index_dict[user_id] # user_index
    mi = tdata.movie_index_dict[movie_id] # movie_index
    user_score_avg = sum(tdata.rate_umat[ui]) / sum(1 for x in tdata.rate_umat[ui] if x > 0)
    predict_value = user_score_avg
    dev_value = 0 # deviation value (if not understand, ask ruicore)
    sum_sim = 0 # (sam as previous line)
    for target_index, sim_value in enumerate(tdata.user_sim_mat[ui]):
        target_user_id = tdata.user_list[target_index]
        if sim_value != 0 and movie_id in tdata.data_udict[target_user_id]:
            target_user_score_avg = sum(tdata.rate_umat[target_index]) / sum(1 for x in tdata.rate_umat[target_index] if x > 0)
            dev_value += (tdata.rate_umat[target_index][mi] - target_user_score_avg) * sim_value
            sum_sim += sim_value
    if sum_sim != 0:
        predict_value += sim_weight*(dev_value / sum_sim)
    return predict_value


def fps_script():
    data_path = 'data-least/ratings.csv'
    rate_list = utils.read_csv(data_path, show_detail=False, shuffle=True)[1:]
    train_data, test_data = utils.split_data(rate_list, 1, show_detail=False)
    tdata = data.TrainData(train_data, show_detail=True, only_hot=False)
    tdata.get_rate_mat()
    tdata.get_fuzzy_mat()
    tdata.get_user_sim(fuzzy_mod=FUZZY_MOD)

    sum_mae = 0
    for index, rate in enumerate(test_data[:TEST_SIZE]):
        predict_value = user_based_model(rate[0], rate[1], tdata, SIM_WEIGHT)
        sum_mae += abs(predict_value - float(rate[2]))
        time_str = time.strftime("%H:%M:%S", time.localtime())
        print('-time: {} - index: {} pre: {:.2} - real: {}'.format(time_str, index, predict_value, rate[2]))

    if TEST_SIZE is None:
        maeLoss = float(sum_mae / len(test_data))
    else:
        maeLoss = float(sum_mae / TEST_SIZE)
    print('--- FUZZY_MOD: {} SIM_WEIGHT: {} TEST_SIZE: {} ---'.format(FUZZY_MOD, SIM_WEIGHT, TEST_SIZE))
    print('maeLoss: {:.2}'.format(maeLoss))

if __name__ == "__main__":
    fps_script()
