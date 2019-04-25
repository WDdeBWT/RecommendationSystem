import time


class RecmModel:

    def __init__(self, train_data, show_detail):
        self.data_udict = {} # {userId: [movieId, ]}
        self.data_idict = {} # {movieID: [userId, ]}
        self.sim_table = {} # {userId: [sim_user, ]}
        self.show_detail = show_detail

        index = 0
        Percentage = 0
        len_tdata = len(train_data)
        for rate in train_data:
            # rate: [userId,movieId,rating,timestamp]
            if index % (len_tdata // 100) == 0:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                if self.show_detail:
                    print('---time: ' + time_str + ' - init_model: ' + str(Percentage) + '% ' + str(index) + '/' + str(len_tdata))
                Percentage += 1
            if rate[0] not in self.data_udict:
                self.data_udict[rate[0]] = []
            self.data_udict[rate[0]].append(rate[1])
            if rate[1] not in self.data_idict:
                self.data_idict[rate[1]] = []
            self.data_idict[rate[1]].append(rate[0])
            index += 1


class RandomModel(RecmModel):

    def get_sim(self, sim_num):
        pass

    def get_recm(self, recm_num):
        recm_table = {} # {userId: [recm_movie, ]}
        for user_id in self.data_udict:
            for i in range(recm_num):
                pass


class UserBasedModel(RecmModel):

    def get_sim(self, sim_num):
        index = 0
        Percentage = 0
        len_udict = len(self.data_udict)
        sim_table = {}
        for user_id in self.data_udict:
            if index % (len_udict // 100) == 0:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                if self.show_detail:
                    print('---time: ' + time_str + ' - get_sim: ' + str(Percentage) + '% ' + str(index) + '/' + str(len_udict))
                Percentage += 1
            sim_table[user_id] = {}
            for movie_id in self.data_udict[user_id]:
                for sim_user in self.data_idict[movie_id]:
                    if sim_user not in sim_table[user_id]:
                        sim_table[user_id][sim_user] = 1
                    else:
                        sim_table[user_id][sim_user] += 1
            # top sim_num similar user
            sim_table[user_id] = [item[0] for item in sorted(sim_table[user_id].items(), key=lambda hit: hit[1], reverse=True)][:sim_num]
            index += 1
        self.sim_table = sim_table

    def get_recm(self, recm_num):
        index = 0
        Percentage = 0
        len_table = len(self.sim_table)
        recm_table = {} # {userId: [recm_movie, ]}
        for user_id in self.sim_table:
            if index % (len_table // 100) == 0:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                if self.show_detail:
                    print('---time: ' + time_str + ' - get_recm: ' + str(Percentage) + '% ' + str(index) + '/' + str(len_table))
                Percentage += 1
            recm_table[user_id] = {}
            for sim_user in self.sim_table[user_id]:
                for movie_id in self.data_udict[sim_user]:
                    if movie_id not in self.data_udict[user_id]:
                        if movie_id not in recm_table[user_id]:
                            recm_table[user_id][movie_id] = 1
                        else:
                            recm_table[user_id][movie_id] += 1
            # top recm_num recommended movie
            recm_table[user_id] = [item[0] for item in sorted(recm_table[user_id].items(), key=lambda hit: hit[1], reverse=True)][:recm_num]
            index += 1
        return recm_table
