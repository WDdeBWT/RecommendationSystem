import os
import csv
import time
import random


def read_csv(r_path):
    content_list = []
    with open(r_path) as r:
        # print('Start read_csv... ', end='', flush=True)
        print('Start read_csv... ', flush=True)
        reader = csv.reader(r)
        index = 0
        for row in reader:
            index += 1
            content_list.append(row)
            if index % 1000000 == 0:
                print('Read index: ' + str(index))
        print('finish')
    return content_list


def write_csv(w_path, content_list):
    with open(w_path, 'w', newline="") as w:
        print('Start write_csv... ', end='')
        writer = csv.writer(w)
        if content_list:
            for line in content_list:
                writer.writerow(line)
        print('finish')


def split_data(rate_list, round):
    train_data = []
    test_data = []
    print('Start split_data... ', end='', flush=True)
    for rate in rate_list:
        if random.randint(1,10) == round:
            test_data.append(rate)
        else:
            train_data.append(rate)
    print('finish')
    return train_data, test_data
