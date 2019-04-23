import math

class Evaluator:

    def __init__(self, test_data, N):
        self.test_data_udict = {}
        self.N = N

        for rate in test_data:
            # rate: [userId,movieId,rating,timestamp]
            if rate[0] not in self.test_data_udict:
                self.test_data_dict[rate[0]] = []
            rate_dict[rate[0]].append(rate[1])
            #{userId: [movieID, ]}

    def Recall(recm_dict):
        hit = 0
        all = 0
        for user_id in recm_dict:
            recm_list = recm_dict[user_id]
            for item in recm_list:
                if item in self.test_data_udict[user_id]:
                    hit += 1
            all += len(self.test_data_udict[user_id])
        return hit / (all * 1.0)


    def Precision(recm_dict):
        hit = 0
        all = 0
        for user_id in recm_dict:
            recm_list = recm_dict[user_id]
            for item in recm_list:
                if item in self.test_data_udict[user_id]:
                    hit += 1
            all += self.N
        return hit / (all * 1.0)


    def Coverage(recm_dict, rate_list):
        recommend_items = set()
        all_items = set()
        for rate in rate_list:
            # rate: [userId,movieId,rating,timestamp]
            all_items.add(rate[1])
        for user_id in recm_dict:
            recm_list = recm_dict[user_id]
            for item in recm_list:
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)


    def Popularity(recm_dict, rate_list):
        item_popularity = dict()
        for rate in rate_list:
            # rate: [userId,movieId,rating,timestamp]
            if rate[1] not in item_popularity:
                item_popularity[rate[1]] = 0
            item_popularity[rate[1]] += 1
        ret = 0
        n = 0
        for user_id in recm_dict:
            recm_list = recm_dict[user_id]
            for item in recm_list:
                ret += math.log(1 + item_popularity[item])
                n += 1
        ret /= n * 1.0
        return ret