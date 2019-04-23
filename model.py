class UserBasedModel:
    def __init__(self, train_data):
        self.data_udict = {} # {userId: [movieId, ]}
        self.data_idict = {} # {movieID: [userId, ]}
        self.sim_table = {} # {userId: [sim_user, ]}

        for rate in train_data:
            # rate: [userId,movieId,rating,timestamp]
            if rate[0] not in self.data_udict:
                self.data_udict[rate[0]] = []
            self.data_udict[rate[0]].append(rate[1])
            if rate[1] not in self.data_idict:
                self.data_idict[rate[1]] = []
            self.data_idict[rate[1]].append(rate[0])

    def get_sim(self, sim_num):
        sim_table = {}
        for user_id in self.data_udict:
            sim_table[user_id] = {}
            for movie_id in self.data_udict[user_id]:
                for sim_user in self.data_idict[movie_id]:
                    if sim_user not in sim_table[user_id]:
                        sim_table[user_id][sim_user] = 1
                    else:
                        sim_table[user_id][sim_user] += 1
            # top sim_num similar user
            sim_table[user_id] = [item[0] for item in sorted(sim_table[user_id].items(), key=lambda hit: hit[1], reverse=True)][:sim_num]
        self.sim_table = sim_table

    def get_recm(self, recm_num):
        recm_table = {} # {userId: [recm_movie, ]}
        for user_id in self.sim_table:
            recm_table[user_id] = {}
            for sim_user in self.sim_table[user_id]:
                for movie_id in self.data_udict[sim_user]:
                    if movie_id not in recm_table[user_id]:
                        recm_table[user_id][movie_id] = 1
                    else:
                        recm_table[user_id][movie_id] += 1
            # top recm_num recommended movie
            recm_table[user_id] = [item[0] for item in sorted(recm_table[user_id].items(), key=lambda hit: hit[1], reverse=True)][:recm_num]
        return recm_table
