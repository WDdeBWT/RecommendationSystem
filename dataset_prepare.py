import os
import time

import utils
import model
import evaluator

data_path = '/Users/baiwentao/Code/RecommendationSystem/data-full/ratings.csv'
write_path = '/Users/baiwentao/Code/RecommendationSystem/data-sub/'

rate_list = utils.read_csv(data_path)
small_list = [rate_list[0]]
medium_list = [rate_list[0]]
big_list = [rate_list[0]]

for rate in rate_list[1:]:
    if int(rate[0]) < 5000 and int(rate[1]) < 2000:
        small_list.append(rate)
    if int(rate[0]) < 10000 and int(rate[1]) < 4000:
        medium_list.append(rate)
    if int(rate[0]) < 20000 and int(rate[1]) < 8000:
        big_list.append(rate)
print(len(small_list)) # 351861
print(len(medium_list)) # 1049943
print(len(big_list)) # 2538594
utils.write_csv(write_path + 'small_list.csv', small_list)
utils.write_csv(write_path + 'medium_list.csv', medium_list)
utils.write_csv(write_path + 'big_list.csv', big_list)
