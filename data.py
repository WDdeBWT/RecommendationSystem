import time

import numpy as np

import utils


class TrainData:

    def __init__(self, train_data, show_detail = False):
        self.data_udict = {} # {userId: {movieId: rate}}
        self.data_idict = {} # {movieID: {userId: rate}
        self.rate_mat = None
        self.show_detail = show_detail
        self._build_init(train_data)

    def _build_init(self, train_data):
        index = 0
        Percentage = 0
        len_tdata = len(train_data)
        for rate in train_data:
            # rate: [userId,movieId,rating,timestamp]
            if self.show_detail:
                if index % (len_tdata // 100) == 0:
                    time_str = time.strftime("%H:%M:%S", time.localtime())
                    if Percentage % 10 == 0:
                        print('---time: ' + time_str + ' - init_model: ' + str(Percentage) + '% ' + str(index) + '/' + str(len_tdata))
                    Percentage += 1
            if rate[0] not in self.data_udict:
                self.data_udict[rate[0]] = {}
            self.data_udict[rate[0]][rate[1]] = rate[2]

            if rate[1] not in self.data_idict:
                self.data_idict[rate[1]] = {}
            self.data_idict[rate[1]][rate[0]] = rate[2]
            index += 1

    def get_rate_mat(self):
        user_list = list(self.data_udict.keys())
        movie_list = list(self.data_idict.keys())
        rate_mat = np.zeros((len(user_list), len(movie_list)))
        for index, user_id in enumerate(user_list):
            if (index + 1) % (len(user_list) // 100) == 0 and self.show_detail:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('---time:{} - TrainData.get_rate_mat {}/{}'.format(time_str, str(index+1), str(len(user_list))))
            for movie_id in self.data_udict[user_id]:
                rate_mat[user_list.index(user_id)][movie_list.index(movie_id)] = self.data_udict[user_id][movie_id]
        self.rate_mat = rate_mat
