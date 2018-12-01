import json
import numpy as np
import torch
import sys
import tagger1_utils as utils
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as functional
import torch.optim as optim

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}

# Globals
EMBEDDING_ROW_LENGTH = 50
WINDOWS_SIZE = 5
EDIM = WINDOWS_SIZE * EMBEDDING_ROW_LENGTH
HID = 100
BATCH = 1024
EPOCHS = 3
LR = 0.01
SEPARATOR = ""


class Trainer(object):
    """
    class trainer in charge of the routine of this app.
    """
    def __init__(self, train, dev, test, model, optimizer, tags_type):
        """
        constructor for Trainer
        :param train: train loader
        :param dev: dev loader
        :param test: test loader
        :param model: model of neural network
        :param optimizer: optimizer from optim
        :param tags_type: tags type - pos or ner
        """
        self.train_loader = train
        self.dev_loader = dev
        self.test_loader = test
        self.model = model
        self.optimizer = optimizer
        self.tags_type = tags_type

    def run(self):
        """
        run routine
        :return: the predicted tags
        """
        dev_avg_loss_in_epoch = {}
        dev_acc_per_epoch = {}
        for epoch in range(1, EPOCHS + 1):
            print "epoch #{}".format(epoch)

            train_loss, correct_train, total_train, train_acc = self.train()
            print "Train: average train loss is {:.4f}, accuracy is {} of {}, {:.00f}%".format(train_loss, correct_train,
                                                                                               total_train, train_acc)

            dev_loss, dev_correct, dev_total, dev_accuracy = self.dev(self.tags_type)
            dev_acc_per_epoch[epoch] = dev_accuracy
            dev_avg_loss_in_epoch[epoch] = dev_loss
            print "Dev: average dev loss is {:.4f}, accuracy is {} of {}, {:.00f}%".format(dev_loss,
                                                                                dev_correct, dev_total, dev_accuracy)

        utils.plot_graph(dev_avg_loss_in_epoch, color="blue", label="Dev Average Loss")
        utils.plot_graph(dev_acc_per_epoch, color="red", label="Dev Average Accuracy")
        tags_predict = self.test(self.tags_type)
        return tags_predict

    def train(self):
        """
        train routine
        :return: train loss, correct, total, acc
        """
        self.model.train()
        train_loss = 0
        correct = 0

        for data, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            loss = functional.nll_loss(output, labels)
            train_loss += loss
            loss.backward()
            self.optimizer.step()

        train_loss /= (len(self.train_loader))
        total = len(self.train_loader) * BATCH
        acc = (100. * correct) / total
        return train_loss, correct, total, acc

    def dev(self, tagger_type):
        """
        dev routine
        :param tagger_type: tagger type ner or pos
        :return: dev loss, correct, total and acc
        """
        self.model.eval()
        validation_loss = 0
        correct = 0
        total = 0
        for data, target in self.dev_loader:
            output = self.model(data)
            validation_loss += functional.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            if tagger_type == 'ner':
                if utils.I2T[pred.cpu().sum().item()] != 'O' or utils.I2T[target.cpu().sum().item()] != 'O':
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total += 1
            else:
                total += 1
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        validation_loss /= len(self.dev_loader)
        accuracy = (100. * correct) / total
        return validation_loss, correct, total, accuracy

    def test(self, tagger_type):
        """
        test routine
        :param tagger_type:
        :return: the predicted tags
        """
        self.model.eval()
        tags_predict = []
        for data in self.test_loader:
            output = self.model(torch.LongTensor(data))
            pred = output.data.max(1, keepdim=True)[1]
            tags_predict.append(pred.item())
        tags_predict = utils.from_index_to_tag(tags_predict)
        return tags_predict


class ComputationGraph(nn.Module):
    """
    Computation graph for neural network
    """
    def __init__(self):
        super(ComputationGraph, self).__init__()

        self.E = nn.Embedding(len(utils.WORDS), EMBEDDING_ROW_LENGTH)
        self.input_size = EDIM
        self.layer0 = nn.Linear(EDIM, HID)
        self.layer1 = nn.Linear(HID, len(utils.TAGS))

    def forward(self, x):
        x = self.E(x).view(-1, self.input_size)
        x = torch.tanh(self.layer0(x))
        x = self.layer1(x)
        return functional.log_softmax(x, dim=1)


# For train and dev
def get_data_as_windows(data_file, is_train=True, separator=" "):
    """
    get data as windows saved in data loader
    :param data_file: path to file
    :param is_train: is train routine
    :param separator: seprator in files
    :return: data loader
    """
    print "Getting data from: ", data_file
    sentences = utils.read_data(data_file, is_train=True, seperator=separator)
    if is_train:
        utils.initialize_indexes()
    windows, tags = utils.create_windows(sentences)
    windows, tags = np.asarray(windows, np.float32), np.asarray(tags, np.int32)
    windows, tags = torch.from_numpy(windows), torch.from_numpy(tags)
    windows, tags = windows.type(torch.LongTensor), tags.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(windows, tags)
    if is_train:
        return DataLoader(dataset,batch_size=BATCH, shuffle=True)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_loader_for_test(data_file):
    """
    get loader for the test routine
    :param data_file:
    :return: loader for test
    """
    print "Getting data from: ", data_file
    sentences = utils.read_data(data_file, tagged_data=False, is_train=False)
    windows = utils.create_windows_without_tags(sentences)
    return windows


def write_result(input_test_path, output_path, tags):
    """
    writes the result to the file
    :param input_test_path: input file
    :param output_path: output path
    :param tags: predicted tags
    :return:
    """
    print "Writing result"
    out_fd = open(output_path, 'w')
    in_fd = open(input_test_path, 'r')
    tags_index = 0
    for line in in_fd:
        if line == '\n':
            out_fd.write(line)
        else:
            out_fd.write("{}{}{}\n".format(line.strip(),SEPARATOR, tags[tags_index]))
            tags_index += 1


def routine(tags_type):
    """
    routine of this app creating the parameters for the trainer and runs it.
    :param tags_type:
    :return:
    """
    train_file, dev_file, test_file = tags_type + "/" + "train", tags_type + "/" + "dev", tags_type + "/" + "test"
    # Create loaders
    train = get_data_as_windows(train_file, separator=SEPARATOR)
    dev = get_data_as_windows(dev_file, is_train=False, separator=SEPARATOR)
    test = get_loader_for_test(test_file)

    model = ComputationGraph()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainer = Trainer(train, dev, test, model, optimizer, tags_type)
    tags_predict = trainer.run()
    write_result(test_file, "test1."+tags_type, tags_predict)


def initialize_globals(tags_type):
    """
    initialize the globals of this app from json file
    :param tags_type:
    :return:
    """
    config_fd = open("taggers_params.json", "r")
    data = json.load(config_fd)
    config_fd.close()
    config = data[tags_type]
    global LR, EPOCHS, BATCH, SEPARATOR
    SEPARATOR = config["SEPARATOR"]
    LR = config["LR"]
    BATCH = config["BATCH"]
    EPOCHS = config["EPOCHS"]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(-1)
    else:
        tags_type = sys.argv[1]
        initialize_globals(tags_type)
        routine(tags_type)

