from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import dynet_config
dynet_config.set(random_seed=0)
import dynet as dy
import numpy as np

EMBEDDING_ROW_LENGTH = 50
WINDOWS_SIZE = 5
INPUT_SIZE = WINDOWS_SIZE * EMBEDDING_ROW_LENGTH
EDIM = 250
HID = 10
NOUT = 10  # size of labels

#
# model = dy.Model()
# pW1 = model.add_parameters((HID, EDIM))
# pb1 = model.add_parameters(HID)
# pW2 = model.add_parameters((NOUT, HID))
# pb2 = model.add_parameters(NOUT)
# E = model.add_lookup_parameters((V, EDIM))
#
#
# def predict_labels(doc):
#     x = encode_doc(doc)
#     h = layer1(x)
#     y = layer2(h)
#     return dy.softmax(y)
#
#
# def encode_doc(doc):
#     doc = [w2i[w] for w in doc]
#     embs = [E[idx] for idx in doc]
#     return dy.esum(embs)
#
#
# def layer1(x):
#     W = dy.parameter(pW1)
#     b = dy.parameter(pb1)
#     return dy.tanh(W*x + b)
#
#
# def layer2(x):
#     W = dy.parameter(pW2)
#     b = dy.parameter(pb2)
#     return dy.tanh(W*x + b)
#
#
# def do_loss(probs, label):
#     label = l2i[label]
#     return -dy.log(dy.pick(probs, label))
#
#
# def classify(doc):
#     dy.renew_cg()
#     probs = predict_labels(doc)
#     vals = probs.npvalue()
#     return i2l[np.argmax(vals)]
#

class MLP(object):
    def __init__(self, model, in_dim, hid_dim, out_dim, non_lin=dy.tanh):
        self._W1 = model.add_parameters((hid_dim, in_dim))
        self._b1 = model.add_parameters(hid_dim)
        self._W2 = model.add_parameters((out_dim, hid_dim))
        self._b2 = model.add_parameters(out_dim)
        self.non_lin = non_lin

    def __call__(self, in_exp):
        W1 = dy.parameter(self._W1)
        b1 = dy.parameter(self._b1)
        W2 = dy.parameter(self._W2)
        b2 = dy.parameter(self._b2)
        g = self.non_lin
        return W2*g(W1*in_exp + b1) + b2


class Module(dy.Model):
    pass



dy.Expression
class NNGraph(dy.ComputationGraph):
    def __init__(self):
        super(NNGraph, self).__init__()
        self.E = model.add_lookup_parameters(model)
        self.input_size = INPUT_SIZE



if __name__ == '__main__':
    model = dy.Model()
    mlp = MLP(model, EDIM, HID, NOUT, dy.tanh)
    RNN = dy.LSTMBuilder(1, EDIM, HID, model)
    s = RNN.initial_state()
    for x in inpt:
        s = s.add_input(x)
    y = dy.softmax(mlp(s.output()))
