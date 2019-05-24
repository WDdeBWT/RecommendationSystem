# fuzzy preference similarity with user based model
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import data
import utils


SHOW_DETAIL = True
TEST_SIZE = 1000 # None means full size

FUZZY_MODE = True
SIM_WEIGHT = 0.8
GROUP_MODE = True
GROUP_DISTANCE = 3
WALK_TIMES = 1000
COLD_NUM = 20
PRECISION_RANGE = 0.8


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
    global SHOW_DETAIL
    global TEST_SIZE
    global FUZZY_MODE
    global SIM_WEIGHT
    global GROUP_MODE
    global GROUP_DISTANCE
    global WALK_TIMES
    global COLD_NUM
    global PRECISION_RANGE
    data_path = 'data-least/ratings.csv'
    rate_list = utils.read_csv(data_path, show_detail=SHOW_DETAIL, shuffle=True)[1:]
    train_data, test_data = utils.split_data(rate_list, 8, show_detail=SHOW_DETAIL)
    tdata = data.TrainData(train_data, show_detail=SHOW_DETAIL, only_hot=False)
    tdata.get_rate_mat()
    tdata.get_fuzzy_mat()
    tdata.get_user_sim(fuzzy_mode=FUZZY_MODE)
    if GROUP_MODE:
        tdata.get_user_group(GROUP_DISTANCE, WALK_TIMES)

    sum_mae = 0
    sum_mse = 0
    sum_hit = 0
    cold_num = 0
    cold_hit = 0
    # plt
    if SHOW_DETAIL:
        x = np.array([])
        y1 = np.array([])
        y2 = np.array([])
        plt.figure(figsize=(10, 5))
        plt.ylim((0, 5.9))
        plt.ion()
        plt.xlabel('test sample')
        plt.ylabel('rate')
        plt.plot(x, y1, color='red', label='predict value')
        plt.plot(x, y2, color='blue', label='real value')
        plt.legend(loc='upper right')
    if TEST_SIZE is None:
        TEST_SIZE = len(test_data)
    for index, rate in enumerate(test_data[:TEST_SIZE]):
        predict_value = user_based_model(rate[0], rate[1], tdata, SIM_WEIGHT)
        sum_mae += abs(predict_value - float(rate[2]))
        sum_mse += abs(predict_value - float(rate[2])) ** 2
        if abs(predict_value - float(rate[2])) < PRECISION_RANGE:
            sum_hit += 1
        if rate[0] in tdata.user_index_dict and sum(1 for x in tdata.rate_umat[tdata.user_index_dict[rate[0]]] if x > 0) <= COLD_NUM:
            cold_num += 1
            if abs(predict_value - float(rate[2])) < PRECISION_RANGE:
                cold_hit += 1
        if SHOW_DETAIL:
            time_str = time.strftime("%H:%M:%S", time.localtime())
            print('-time: {} - index: {} pre: {:.2} - real: {}'.format(time_str, index, predict_value, rate[2]))
            if index < 20:
                x = np.append(x, str(index + 1))
                y1 = np.append(y1, predict_value)
                y2 = np.append(y2, float(rate[2]))
                plt.plot(x, y1, color='red', label='predict value')
                plt.plot(x, y2, color='blue', label='real value')
                plt.draw(); plt.pause(0.05)

    maeLoss = float(sum_mae / TEST_SIZE)
    mseLoss = float(sum_mse / TEST_SIZE)
    hitRate = float(sum_hit / TEST_SIZE)
    coldHitRate = float(cold_hit / cold_num)
    # print('--- FUZZY_MODE: {} GROUP_MODE: {} SIM_WEIGHT: {} GROUP_DISTANCE: {} WALK_TIMES: {} TEST_SIZE: {} ---'.format(FUZZY_MODE, GROUP_MODE, SIM_WEIGHT, GROUP_DISTANCE, WALK_TIMES, TEST_SIZE))
    print('----- FUZZY_MODE: {} GROUP_MODE: {} -----'.format(FUZZY_MODE, GROUP_MODE))
    print('> maeLoss: {:.2} mseLoss: {:.2} hitRate: {:.2} coldHitRate: {:.2}'.format(maeLoss, mseLoss, hitRate, coldHitRate))
    if SHOW_DETAIL:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    fps_script()
