import math

class Evaluator:

    def __init__(self, rate_list, test_data, recm_dict, recm_num):
        self.rate_list = rate_list
        self.test_data_udict = {} # {userId: [movieID, ]}
        self.recm_dict = recm_dict
        self.recm_num = recm_num

        for rate in test_data:
            # rate: [userId,movieId,rating,timestamp]
            if rate[0] not in self.test_data_udict:
                self.test_data_udict[rate[0]] = []
            self.test_data_udict[rate[0]].append(rate[1])

    def recall(self):
        hit = 0
        all = 0
        for user_id in self.recm_dict:
            recm_list = self.recm_dict[user_id]
            for item in recm_list:
                if user_id in self.test_data_udict:
                    if item in self.test_data_udict[user_id]:
                        hit += 1
            if user_id in self.test_data_udict:
                all += len(self.test_data_udict[user_id])
        return hit / (all * 1.0)


    def precision(self):
        hit = 0
        all = 0
        for user_id in self.recm_dict:
            recm_list = self.recm_dict[user_id]
            for item in recm_list:
                if user_id in self.test_data_udict:
                    if item in self.test_data_udict[user_id]:
                        hit += 1
            all += self.recm_num
        return hit / (all * 1.0)


    def coverage(self):
        recommend_items = set()
        all_items = set()
        for rate in self.rate_list:
            # rate: [userId,movieId,rating,timestamp]
            all_items.add(rate[1])
        for user_id in self.recm_dict:
            recm_list = self.recm_dict[user_id]
            for item in recm_list:
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)


    def popularity(self):
        item_popularity = dict()
        for rate in self.rate_list:
            # rate: [userId,movieId,rating,timestamp]
            if rate[1] not in item_popularity:
                item_popularity[rate[1]] = 0
            item_popularity[rate[1]] += 1
        ret = 0
        n = 0
        for user_id in self.recm_dict:
            recm_list = self.recm_dict[user_id]
            for item in recm_list:
                ret += math.log(1 + item_popularity[item])
                n += 1
        ret /= n * 1.0
        return ret