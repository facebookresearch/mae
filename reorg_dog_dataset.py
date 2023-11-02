import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

data_dir = os.path.join('..', 'data', 'dog-breed-identification')

def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
print("ReOrganization Successful!!")