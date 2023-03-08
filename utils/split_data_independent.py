

import os
import csv
import numpy as np
import random

'''
Train:(a,b,e,f)*70%
Valid:(a,b,e,f)*20%
Test:(a,b,e,f)*10%+c+d
'''

Physio_Dir="/Users/liuhongye/BIT/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
out_dir = '/Users/liuhongye/BIT/Adversarial_Attacks_for_SER/data/development-subtaskA/evaluation_setup'

trainvalid_subset = ['training-a','training-b','training-e','training-f']
test_exclusive_subset = ['training-c','training-d']

train_data_dir = os.path.join(out_dir,'train.txt')
valid_data_dir = os.path.join(out_dir,'valid.txt')
test_data_dir = os.path.join(out_dir,'test.txt')
trainvalid_data_dir = os.path.join(out_dir,'traindevel.txt')
total_data_dir = os.path.join(out_dir,'total.csv')

train_name_label = []
valid_name_label = []
test_name_label = []
trainvalid_name_label = []
total_data = []
for subset in trainvalid_subset:
    csv_dir = os.path.join(Physio_Dir,subset,'REFERENCE.csv')
    # read REFERENCE.csv and shuffle to get train,valid,test name,label
    with open(csv_dir, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    random.Random(1234).shuffle(lis)

    train_num = int(len(lis)*0.7)
    valid_num = int(len(lis)*0.2)
    test_num = len(lis)-train_num-valid_num
    train_name_label.extend(lis[:train_num])
    valid_name_label.extend(lis[train_num:train_num+valid_num])

    test_name_label.extend(lis[-test_num:])
    total_data.extend(lis)

trainvalid_name_label.extend(train_name_label)
trainvalid_name_label.extend(valid_name_label)

for subset in test_exclusive_subset:
    csv_dir = os.path.join(Physio_Dir,subset,'REFERENCE.csv')
    # read REFERENCE.csv and shuffle to add test name,label

    with open(csv_dir, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        # random.Random(1234).shuffle(lis)

    test_name_label.extend(lis)
    total_data.extend(lis)

### write a txt file from a list
with open(train_data_dir, 'w') as file:
    for inner_list in train_name_label:
        for item in inner_list:
            file.write("%s\n" % item)

with open(valid_data_dir, 'w') as file:
    for inner_list in valid_name_label:
        for item in inner_list:
            file.write("%s\n" % item)

with open(test_data_dir, 'w') as file:
    for inner_list in test_name_label:
        for item in inner_list:
            file.write("%s\n" % item)

with open(trainvalid_data_dir, 'w') as file:
    for inner_list in trainvalid_name_label:
        for item in inner_list:
            file.write("%s\n" % item)

with open(total_data_dir, 'w') as file:
    for inner_list in total_data:
        for item in inner_list:
            file.write("%s\n" % item)
