"""

.. _l-ortmodule:

Pytorch and onnxruntime
=======================

.. contents::
    :local:

A neural network with scikit-learn
++++++++++++++++++++++++++++++++++

"""
import warnings
from pprint import pprint
import time
import os
import numpy
import onnx
from pandas import DataFrame
from onnxruntime import get_device
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def build_model(N=1000, n_features=5, hidden_layer_sizes="4,3", max_iter=1000,
                learning_rate_init=1e-4, batch_size=100,
                device='cpu', opset=12, profile=True):
    """
    Compares :epkg:`onnxruntime-training` to :epkg:`scikit-learn` for
    training. Training algorithm is SGD.

    :param N: number of observations to train on
    :param n_features: number of features
    :param hidden_layer_sizes: hidden layer sizes, comma separated values
    :param max_iter: number of iterations
    :param learning_rate_init: initial learning rate
    :param batch_size: batch size
    :param run_torch: train scikit-learn in the same condition (True) or
        just walk through one iterator with *scikit-learn*
    :param device: `'cpu'` or `'cuda'`
    :param opset: opset to choose for the conversion
    :param profile: if True, run the profiler on training steps
    """
    N = int(N)
    n_features = int(n_features)
    max_iter = int(max_iter)
    learning_rate_init = float(learning_rate_init)
    batch_size = int(batch_size)
    profile = profile in (1, True, '1', 'True')

    print("N=%d" % N)
    print("n_features=%d" % n_features)
    print("hidden_layer_sizes=%r" % (hidden_layer_sizes, ))
    print("max_iter=%d" % max_iter)
    print("learning_rate_init=%f" % learning_rate_init)
    print("batch_size=%d" % batch_size)
    print("opset=%r (unused)" % opset)
    print("device=%r" % device)
    device0 = device
    device = torch.device(
        "cuda:0" if device in ('cuda', 'cuda:0', 'gpu') else "cpu")
    print("fixed device=%r" % device)
    print('------------------')

    if not isinstance(hidden_layer_sizes, tuple):
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(",")))
    X, y = make_regression(N, n_features=n_features, bias=2)
    X = X.astype(numpy.float32)
    y = y.astype(numpy.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    class Net(torch.nn.Module):
        def __init__(self, n_features, hidden, n_output):
            super(Net, self).__init__()
            self.hidden = []

            size = n_features
            for i, hid in enumerate(hidden_layer_sizes):
                self.hidden.append(torch.nn.Linear(size, hid))
                size = hid
                setattr(self, "hid%d" % i, self.hidden[-1])
            self.hidden.append(torch.nn.Linear(size, n_output))
            setattr(self, "predict", self.hidden[-1])

        def forward(self, x):
            for hid in self.hidden:
                x = hid(x)
                x = F.relu(x)
            return x

    nn = Net(n_features, hidden_layer_sizes, 1)
    if device0 == 'cpu':
        nn.cpu()
    else:
        nn.cuda(device=device)
    print("n_parameters=%d, n_layers=%d" % (
        len(list(nn.parameters())), len(nn.hidden)))
    for i, p in enumerate(nn.parameters()):
        print("  p[%d].shape=%r" % (i, p.shape))

    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate_init)
    criterion = torch.nn.MSELoss(size_average=False)
    batch_no = len(X_train) // batch_size

    # training
    inputs = torch.tensor(
        X_train[:1], requires_grad=True, device=device)
    nn(inputs)

    def train_torch():
        for epoch in range(max_iter):
            running_loss = 0.0
            x, y = shuffle(X_train, y_train)
            for i in range(batch_no):
                start = i * batch_size
                end = start + batch_size
                inputs = torch.tensor(
                    x[start:end], requires_grad=True, device=device)
                labels = torch.tensor(
                    y[start:end], requires_grad=True, device=device)

                def step_torch():
                    optimizer.zero_grad()
                    outputs = nn(inputs)
                    loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
                    loss.backward()
                    optimizer.step()
                    return loss

                loss = step_torch()
                running_loss += loss.item()
        return running_loss

    begin = time.perf_counter()
    running_loss = train_torch()
    dur_torch = time.perf_counter() - begin

    print("time_torch=%r, running_loss=%r" % (dur_torch, running_loss))
    return nn


def save_as_onnx(model, filename, size, target_opset=14, batch_size=1, device='cpu'):
    size = (batch_size, ) + (size, )
    x = torch.randn(size, requires_grad=True).to(device)
    torch.onnx.export(
        model, x, filename,
        training=torch.onnx.TrainingMode.TRAINING,
        do_constant_folding=False,
        export_params=False,
        keep_initializers_as_inputs=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}})


net = build_model()
save_as_onnx(net, "model.onnx", 5)

with open("model.onnx", "rb") as f:
    onx = onnx.load(f)
print(onx)
