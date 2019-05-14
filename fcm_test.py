import time
import math

import numpy as np

import utils
import model_test
import evaluator
import data
import FCM


def test():
    data_path = 'data-least/ratings.csv'
    rate_list = utils.read_csv(data_path, False)[1:]
    train_data, test_data = utils.split_data(rate_list, 1, False)
    tdata = data.TrainData(train_data, show_detail=True, only_hot=False)
    tdata.get_rate_mat()

    fcm_u = FCM.FCM(tdata.rate_umat, 20, 20, show_detail=True)
    user_membership_mat, user_cluster_centers = fcm_u.get_result()
    user_membership_mat = np.array(user_membership_mat)
    user_cluster_centers = np.array(user_cluster_centers)
    print(user_membership_mat.shape)
    print(user_cluster_centers.shape)

    fcm_i = FCM.FCM(tdata.rate_imat, 20, 20, show_detail=True)
    movie_membership_mat, movie_cluster_centers = fcm_i.get_result()
    movie_membership_mat = np.array(movie_membership_mat)
    movie_cluster_centers = np.array(movie_cluster_centers)
    print(movie_membership_mat.shape)
    print(movie_cluster_centers.shape)    


test()
