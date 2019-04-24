import os
import csv
import time
import random

data_path = '/mnt/lustressd/baiwentao/project_temp/RecommendationSystem/data-full/ratings.csv'


def read_csv(r_path):
    content_list = []
    with open(r_path) as r:
        print('Start read file...')
        reader = csv.reader(r)
        print('Read finish, write to content_list...')
        index = 0
        for row in reader:
            if index % 1000000 == 0:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('---time: ' + time_str + ' - read_csv line: ' + str(index))
            content_list.append(row)
            index += 1
    return content_list


def write_csv(w_path, content_list):
    with open(w_path, 'w', newline="") as w:
        writer = csv.writer(w)
        writer.writerow(['Name', 'State'])
        if content_list:
            for line in content_list:
                line[0].strip()
                writer.writerow(line)


def split_data(rate_list, round):
    train_data = []
    test_data = []
    index = 0
    len_rlist = len(rate_list)
    for rate in rate_list:
        if index % 1000000 == 0:
            time_str = time.strftime("%H:%M:%S", time.localtime())
            print('---time: ' + time_str + ' - get_sim: ' + str(index) + '/' + str(len_rlist))
        if random.randint(1,10) == round:
            test_data.append(rate)
        else:
            train_data.append(rate)
        index += 1
    return train_data, test_data
