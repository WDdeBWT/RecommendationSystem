import os
import csv
import time
import random


def read_csv(r_path, show_detail=True):
    content_list = []
    with open(r_path) as r:
        if show_detail:
            # print('Start read_csv... ', end='', flush=True)
            print('Start read_csv... ', flush=True)
        reader = csv.reader(r)
        index = 0
        for row in reader:
            index += 1
            content_list.append(row)
            if index % 1000000 == 0:
                if show_detail:
                    print('Read index: ' + str(index))
        if show_detail:
            print('finish')
    return content_list


def write_csv(w_path, content_list, show_detail=True):
    with open(w_path, 'w', newline="") as w:
        if show_detail:
            print('Start write_csv... ', end='', flush=True)
        writer = csv.writer(w)
        if content_list:
            for line in content_list:
                writer.writerow(line)
        if show_detail:
            print('finish')


def write_log(w_path, content_list):
    with open(w_path, 'a') as w:
        if content_list:
            time_str = time.strftime("%H:%M:%S", time.localtime())
            w.write('\n--- time: ' + time_str + ' ---\n')
            for line in content_list:
                w.write(str(line) + '\n')


def split_data(rate_list, round, show_detail=True):
    train_data = []
    test_data = []
    if show_detail:
        print('Start split_data... ', end='', flush=True)
    for index, rate in enumerate(rate_list):
        # if random.randint(1,10) == round:
        if index % 10 == round:
            test_data.append(rate)
        else:
            train_data.append(rate)
    if show_detail:
        print('finish')
    return train_data, test_data
