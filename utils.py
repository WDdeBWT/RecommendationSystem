import os
import csv
import random

data_path = '/Users/baiwentao/Code/RecommendationSystem/data-least/ratings.csv'


def read_csv(r_path):
    content_list = []
    with open(r_path) as r:
        reader = csv.reader(r)
        for row in reader:
            content_list.append(row)
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
    for rate in rate_list:
        if random.randint(1,10) == round:
            test_data.append(rate)
        else:
            train_data.append(rate)
    return train_data, test_data
