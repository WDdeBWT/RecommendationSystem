import time
import math

import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import numpy as np

import FCM
import data
import utils
import evaluator
import model_test


CLUSTERS = 5


def test():
    data_path = 'data-sub/temp_list.csv'
    rate_list = utils.read_csv(data_path, show_detail=False)[1:]
    train_data, test_data = utils.split_data(rate_list, 1, show_detail=False)
    tdata = data.TrainData(train_data, show_detail=True, only_hot=True)
    # tdata.get_rate_mat()
    tdata.get_user_sim()

    fcm_u = FCM.FCM(tdata.user_sim_mat, CLUSTERS, 20, show_detail=True)
    user_membership_mat = fcm_u.get_result()
    user_membership_mat = np.array(user_membership_mat)
    print('user_membership_mat.shape: {}'.format(user_membership_mat.shape))


if __name__ == "__main__":
    test()
