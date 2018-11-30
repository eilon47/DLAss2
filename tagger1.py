import numpy as np
import torch
import sys
import tagger1_utils as utils
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as functional
import torch.optim as optim

EMBEDDING_ROW_LENGTH = 50
WINDOWS_SIZE = 5
EDIM = WINDOWS_SIZE * EMBEDDING_ROW_LENGTH
HID = 10
NOUT = 10  # size of labels


class Trainer(object):
    pass


class ComputationGraph(nn.Module):
    pass


# For train and dev
def get_data_as_windows(data_file, tagged_file=True, with_tags=True):
    sentences = utils.read_data(data_file, tagged_file)
    windows, tags = utils.create_windows(sentences, with_tags=with_tags)
    windows, tags =



def routine(tags_type):
    train_file, dev_file, test_file = tags_type + "/" + "train", tags_type + "/" + "dev", tags_type + "/" + "test"


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(-1)
    else:
        tags_type = sys.argv[1]

