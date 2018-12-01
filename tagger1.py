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
BATCH = 1024
EPOCHS = 3




class Trainer(object):
    def __init__(self, train, dev, test, model, optimizer, tags_type):
        self.train_loader = train
        self.dev_loader = dev
        self.test_loader = test
        self.model = model
        self.optimizer = optimizer
        self.tags_type = tags_type

    def run(self):
        avg_train_loss_per_epoch_dict = {}
        avg_validation_loss_per_epoch_dict = {}
        validation_accuracy_per_epoch_dict = {}
        for epoch in range(1, EPOCHS + 1):
            print str(epoch)
            self.train(epoch, avg_train_loss_per_epoch_dict)
            self.dev(epoch, avg_validation_loss_per_epoch_dict,
                            validation_accuracy_per_epoch_dict, self.tags_type)
        #plotTrainAndValidationGraphs(avg_validation_loss_per_epoch_dict, validation_accuracy_per_epoch_dict)
        self.test(self.tags_type)

    def train(self, epoch, avg_train_loss_per_epoch_dict):
        """
        go through all examples on the validation set, calculates perdiction, loss
        , accuracy, and updating the model parameters.
        :param epoch: number of epochs
        :param avg_train_loss_per_epoch_dict: avg loss per epoch dictionary
        :return: None
        """
        self.model.train()
        train_loss = 0
        correct = 0

        for data, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            # negative log likelihood loss
            loss = F.nll_loss(output, labels)
            train_loss += loss
            # calculating gradients
            loss.backward()
            # updating parameters
            self.optimizer.step()

        train_loss /= (len(self.train_loader))
        avg_train_loss_per_epoch_dict[epoch] = train_loss
        print("Epoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)".format(epoch, train_loss, correct,
                                                                                             len(
                                                                                                 self.train_loader) * BATCH,
                                                                                             100. * correct / (len(
                                                                                                 self.train_loader) * BATCH)))

    def dev(self, epoch_num, avg_validation_loss_per_epoch_dict,
                   validation_accuracy_per_epoch_dict, tagger_type):
        """
        go through all examples on the validation set, calculates perdiction, loss
        and accuracy
        :param epoch: number of epochs
        :param avg_train_loss_per_epoch_dict: avg loss per epoch dictionary
        :return: None
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
        avg_validation_loss_per_epoch_dict[epoch_num] = validation_loss
        accuracy = 100. * correct / total
        validation_accuracy_per_epoch_dict[epoch_num] = accuracy

        print('\n Epoch:{} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, validation_loss, correct, total,
            accuracy))

    def test(self, tagger_type):
        """
        writes all the model predictions on the test set to test.pred file.
        :return:  None
        """
        self.model.eval()
        pred_list = []
        for data in self.test_loader:
            output = self.model(torch.LongTensor(data))
            # get the predicted class out of output tensor
            pred = output.data.max(1, keepdim=True)[1]
            # add current prediction to predictions list
            pred_list.append(pred.item())

        pred_list = self.convert_tags_indices_to_tags(pred_list)
        self.write_test_results_file(tagger_type + "/test", "test1." + tagger_type, pred_list)

    def convert_tags_indices_to_tags(self, tags_indices_list):
        """
        Converts list of tags indices to tags list (string representation).
        :param tags_indices_list: tags indices list
        :return: tags list (string representation)
        """
        return [utils.I2T[index] for index in tags_indices_list]

    def write_test_results_file(self, test_file_name, output_file_name, predictions_list):
        """
        writes test predictions to output file.
        :param test_file_name: test file name
        :param output_file_name: output file name
        :param predictions_list: predictions for every word in the test data.
        """
        with open(test_file_name, 'r') as test_file, open(output_file_name, 'w') as output:
            content = test_file.readlines()
            i = 0
            for line in content:
                if line == '\n':
                    output.write(line)
                else:
                    output.write(line.strip('\n') + " " + predictions_list[i] + "\n")
                    i += 1


class ComputationGraph(nn.Module):
    def __init__(self):
        super(ComputationGraph, self).__init__()

        self.E = nn.Embedding(len(utils.WORDS), EMBEDDING_ROW_LENGTH)
        self.input_size = EDIM
        self.layer0 = nn.Linear(EDIM, HID)
        self.layer1 = nn.Linear(HID, len(utils.TAGS))

    def forward(self, x):
        x = self.E(x).view(-1, self.input_size)
        x = functional.tanh(self.layer0(x))
        x = self.layer1(x)
        return functional.log_softmax(x, dim=1)


# For train and dev
def get_data_as_windows(data_file, is_train=True):
    print "Getting data from: ", data_file
    sentences = utils.read_data(data_file, is_train=True)
    windows, tags = utils.create_windows(sentences)
    windows, tags = np.asarray(windows, np.float32), np.asarray(tags, np.int32)
    windows, tags = torch.from_numpy(windows), torch.from_numpy(tags)
    windows, tags = windows.type(torch.LongTensor), tags.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(windows, tags)
    if is_train:
        return DataLoader(dataset,batch_size=BATCH, shuffle=True)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_loader_for_test(data_file):
    print "Getting data from: ", data_file
    sentences = utils.read_data(data_file, tagged_data=False, is_train=False)
    windows = utils.create_windows_without_tags(sentences)
    return windows


def routine(tags_type):
    train_file, dev_file, test_file = tags_type + "/" + "train", tags_type + "/" + "dev", tags_type + "/" + "test"
    train = get_data_as_windows(train_file)
    dev = get_data_as_windows(dev_file, is_train=False)
    test = get_loader_for_test(test_file)
    model = ComputationGraph()
    optimizer = optim.Adam(model.parameters())
    trainer = Trainer(train, dev, test, model, optimizer, tags_type)
    trainer.run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(-1)
    else:
        tags_type = sys.argv[1]
        routine(tags_type)

