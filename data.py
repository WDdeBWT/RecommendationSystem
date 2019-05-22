import time
import random

import numpy as np

import utils


class TrainData:

    def __init__(self, train_data, show_detail = False, only_hot = False):
        self.data_udict = {} # {userId: {movieId: rate}}
        self.data_idict = {} # {movieID: {userId: rate}
        self.rate_umat = None
        self.rate_imat = None
        self.user_list = None
        self.movie_list = None
        self.user_index_dict = None
        self.movie_index_dict = None
        self.fuzzy_mat = None
        self.user_sim_mat = None
        self.show_detail = show_detail
        self.only_hot = only_hot
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
                        print('-time: ' + time_str + ' - init_model: ' + str(Percentage) + '% ' + str(index) + '/' + str(len_tdata))
                    Percentage += 1
            if rate[0] not in self.data_udict:
                self.data_udict[rate[0]] = {}
            self.data_udict[rate[0]][rate[1]] = float(rate[2])

            if rate[1] not in self.data_idict:
                self.data_idict[rate[1]] = {}
            self.data_idict[rate[1]][rate[0]] = float(rate[2])
            index += 1
        if self.only_hot:
            delete_users = []
            delete_movies = []
            for user_id in list(self.data_udict.keys()):
                if len(self.data_udict[user_id]) < 10:
                    delete_users.append(user_id)
            for movie_id in list(self.data_idict.keys()):
                if len(self.data_idict[movie_id]) < 10:
                    delete_movies.append(movie_id)
            for user_id in delete_users:
                self.data_udict.pop(user_id)
                for movie_id in self.data_idict:
                    if user_id in self.data_idict[movie_id]:
                        self.data_idict[movie_id].pop(user_id)
            for movie_id in delete_movies:
                self.data_idict.pop(movie_id)
                for user_id in self.data_udict:
                    if movie_id in self.data_udict[user_id]:
                        self.data_udict[user_id].pop(movie_id)

        self.user_list = list(self.data_udict.keys())
        self.movie_list = list(self.data_idict.keys())
        self.user_index_dict = {user_id: index for index, user_id in enumerate(self.user_list)}
        self.movie_index_dict = {movie_id: index for index, movie_id in enumerate(self.movie_list)}

    def get_rate_mat(self):
        rate_umat = np.zeros((len(self.user_list), len(self.movie_list)))
        rate_imat = np.zeros((len(self.movie_list), len(self.user_list)))
        for index, user_id in enumerate(self.user_list):
            if (index + 1) % (len(self.user_list) // 10) == 0 and self.show_detail:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('-time: {} - TrainData.get_rate_mat {}/{}'.format(time_str, str(index+1), str(len(self.user_list))))
            for movie_id in self.data_udict[user_id]:
                rate_umat[self.user_index_dict[user_id]][self.movie_index_dict[movie_id]] = self.data_udict[user_id][movie_id]
                rate_imat[self.movie_index_dict[movie_id]][self.user_index_dict[user_id]] = self.data_udict[user_id][movie_id]
        self.rate_umat = rate_umat
        self.rate_imat = rate_imat

    def get_fuzzy_mat(self):

        # def get_fuzzy_rate(max_score, inp_score):
        #     rate = np.zeros(3)
        #     mid_score = max_score / 2
        #     rate[2] = inp_score/max_score
        #     rate[1] = 1 - abs(inp_score - mid_score) / mid_score
        #     rate[0] = 1 - rate[2]
        #     return rate/sum(rate)

        def get_fuzzy_rate(max_score, inp_score):
            mid_score = max_score * 0.6
            rate = np.zeros(3)
            if inp_score > mid_score:
                rate[2] = (inp_score - mid_score) / max_score - mid_score
            else:
                rate[0] = (mid_score - inp_score) / mid_score
            rate[1] = 1 - rate[0] - rate[2]
            return rate

        fuzzy_mat = np.zeros((len(self.user_list), len(self.movie_list), 3)) # shape = user * item * class
        for index, user_id in enumerate(self.user_list):
            if (index + 1) % (len(self.user_list) // 10) == 0 and self.show_detail:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('-time: {} - TrainData.get_fuzzy_mat {}/{}'.format(time_str, str(index+1), str(len(self.data_udict))))
            for movie_id in self.data_udict[user_id]:
                fuzzy_rate = get_fuzzy_rate(5, self.data_udict[user_id][movie_id])
                for i in range(3):
                    fuzzy_mat[self.user_index_dict[user_id]][self.movie_index_dict[movie_id]][i] = fuzzy_rate[i]
        self.fuzzy_mat = fuzzy_mat

    def get_user_sim(self, fuzzy_mode=True):
        def get_cos_sim(v1, v2):
            num = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            return num / denom

        def gen_pcc_sim(u_avg, v_avg, urates, vrates):
            if not urates or not vrates: return 0.00
            u_diff = [x - u_avg for x in urates]
            v_diff = [x - v_avg for x in vrates]
            a = sum(x * y for x, y in zip(u_diff, v_diff))
            b = (sum(x**2 for x in u_diff) * sum(y**2 for y in v_diff))**0.5
            return a / b if b else 0.00

        user_sim_mat = np.zeros((len(self.user_list), len(self.user_list)))
        for index, user_id in enumerate(self.user_list):
            if (index + 1) % (len(self.user_list) // 10) == 0 and self.show_detail:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('-time: {} - TrainData.get_user_sim {}/{}'.format(time_str, str(index+1), str(len(self.data_udict))))
            ui = self.user_index_dict[user_id] # user_index
            for target_user_id in self.user_list:
                ti = self.user_index_dict[target_user_id] # target_user_index
                if ui == ti:
                    continue
                if fuzzy_mode:
                    common_count = 0
                    for movie_id in self.movie_list:
                        if movie_id in self.data_udict[user_id] and movie_id in self.data_udict[target_user_id]:
                            mi = self.movie_index_dict[movie_id] # movie_index
                            # user_sim_mat[ui][ti] += get_cos_sim(self.fuzzy_mat[ui][mi], self.fuzzy_mat[ti][mi])
                            user_sim_mat[ui][ti] += gen_pcc_sim(3, 3, self.fuzzy_mat[ui][mi].tolist(), self.fuzzy_mat[ti][mi].tolist())
                            common_count += 1
                    if common_count != 0:
                        user_sim_mat[ui][ti] /= common_count
                else:
                    user_hit_list = []
                    target_user_hit_list = []
                    for movie_id in self.movie_list:
                        if movie_id in self.data_udict[user_id] and movie_id in self.data_udict[target_user_id]:
                            user_hit_list.append(self.data_udict[user_id][movie_id])
                            target_user_hit_list.append(self.data_udict[target_user_id][movie_id])
                    if len(user_hit_list) != 0:
                        user_sim_mat[ui][ti] += get_cos_sim(np.array(user_hit_list), np.array(target_user_hit_list))
            if sum(user_sim_mat[self.user_index_dict[user_id]]) != 0:
                user_sim_mat[self.user_index_dict[user_id]] /= sum(user_sim_mat[self.user_index_dict[user_id]])
        self.user_sim_mat = user_sim_mat

    def get_user_group(self, distance, walk_times = 1000):
        # user group member weight replace sim value
        new_sim_mat = np.zeros((len(self.user_list), len(self.user_list)))
        for index, user_id in enumerate(self.user_list):
            if (index + 1) % (len(self.user_list) // 10) == 0 and self.show_detail:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('-time: {} - TrainData.get_user_group {}/{}'.format(time_str, str(index+1), str(len(self.user_list))))
            group = {} # {index: hit_times}
            for i in range(walk_times):
                crt_user = index
                for i in range(distance):
                    sum_p = 0 # sum probability
                    tar_p = random.random() # target probability
                    for target_user_index in range(len(self.user_list)):
                        sum_p += self.user_sim_mat[crt_user][target_user_index]
                        if sum_p > tar_p:
                            if crt_user not in group:
                                group[crt_user] = 1
                            else:
                                group[crt_user] += 1
                            crt_user = target_user_index
                            break
            for item in group.items():
                new_sim_mat[index][item[0]] = item[1] / (distance * walk_times)
        self.user_sim_mat = new_sim_mat


if __name__ == "__main__":
    data_path = 'data-sub/temp_list.csv'
    rate_list = utils.read_csv(data_path, show_detail=False)[1:]
    tdata = TrainData(rate_list, show_detail=False, only_hot=False)
    a = tdata.data_idict[tdata.movie_list[10]]
    print(tdata.movie_list[10])
    for i in a.items():
        print(i)
