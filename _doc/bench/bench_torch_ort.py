"""
.. _l-bench-torch-ort-factory:

Looks into the time spent in function while using ONNX to extend pytorch
========================================================================


.. contents::
    :local:

ONNX graph
++++++++++

"""
import os
import pprint
import logging
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxSigmoid, OnnxMatMul, OnnxAdd)
from pyquickhelper.pycode.profiling import profile, profile2graph
from mlprodict.onnx_tools.onnx_manipulations import onnx_rename_names
from mlprodict.plotting.plotting_onnx import plot_onnx
import torch
from deeponnxcustom.experimental.torchort import TorchOrtFactory


def from_numpy(v, device=None, requires_grad=False):
    """
    Convers a numpy array into a torch array and
    sets *device* and *requires_grad*.
    """
    v = torch.from_numpy(v)
    if device is not None:
        v = v.to(device)
    v.requires_grad_(requires_grad)
    return v


def create_onnx_graph(N, d_in=3, d_out=2, n_loops=1, opv=14):
    """
    Returns a weird ONNX graph and its weights.
    """
    var = [('X', FloatTensorType([N, d_in]))]

    sum_node = None
    weights_values = []
    for i in range(n_loops):
        cst = numpy.random.randn(d_in, 1).astype(numpy.float32) / (i + 1)
        weights_values.append(cst)
        mul = OnnxMatMul(var[0], cst, op_version=opv)
        tanh = OnnxSigmoid(mul, op_version=opv)
        if sum_node is None:
            sum_node = tanh
        else:
            sum_node = OnnxAdd(sum_node, tanh, op_version=opv)

    cst_mul = numpy.random.randn(1, d_out).astype(numpy.float32)
    weights_values.append(cst_mul)
    mul = OnnxMatMul(sum_node, cst_mul, op_version=opv)

    cst_add = numpy.random.randn(1, d_out).astype(numpy.float32)
    weights_values.append(cst_add)
    final = OnnxAdd(mul, cst_add, op_version=opv, output_names=['Y'])

    onx = final.to_onnx(
        var, target_opset=opv, outputs=[('Y', FloatTensorType())])

    weights_name = [i.name for i in onx.graph.initializer]
    new_names = ['W%03d' % i for i in range(len(weights_name))]
    onx = onnx_rename_names(onx, replace=dict(zip(weights_name, new_names)))
    weights = list(zip(new_names, weights_values))
    return onx, weights


N, d_in, d_out = 5, 3, 2
enable_logging = False
onx, weights = create_onnx_graph(N, n_loops=20)

with open("bench_torch_ort.onnx", "wb") as f:
    f.write(onx.SerializeToString())


###############################################
# Wraps ONNX as a torch.autograd.Function
# +++++++++++++++++++++++++++++++++++++++

fact = TorchOrtFactory(onx, [w[0] for w in weights])
cls = fact.create_class(keep_models=True, enable_logging=False)
print(cls)


##########################################
# Training
# ++++++++


def train_cls(cls, device, x, y, weights, n_iter=100, learning_rate=1e-2):
    x = from_numpy(x, requires_grad=True, device=device)
    y = from_numpy(y, requires_grad=True, device=device)

    weights_tch = [(w[0], from_numpy(w[1], requires_grad=True, device=device))
                   for w in weights]
    weights_values = [w[1] for w in weights_tch]

    all_losses = []
    for t in range(n_iter):
        # forward - backward
        y_pred = cls.apply(x, *weights_values)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()

        # update weights
        with torch.no_grad():
            for name, w in weights_tch:
                w -= w.grad * learning_rate
                w.grad.zero_()

        all_losses.append((t, float(loss.detach().numpy())))
    return all_losses, weights_tch


device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("device:", device)

x = numpy.random.randn(N, d_in).astype(numpy.float32)
y = numpy.random.randn(N, d_out).astype(numpy.float32)
train_losses, final_weights = train_cls(cls, device, x, y, weights, n_iter=5)

pprint.pprint(train_losses)

#######################################
# Profiling
# +++++++++

folder = os.path.abspath(os.getcwd()).split('deeponnxcustom')[0]
folder2 = os.path.abspath(os.path.split(
    os.path.dirname(torch.__file__))[0])[:-6]

ps, text = profile(lambda: train_cls(cls, device, x, y, weights, n_iter=100))
print(type(ps))
print(text.replace(folder, "").replace(folder2, ""))

########################################
# Other presentation
# ++++++++++++++++++

folder = folder.replace("\\", "/")
folder2 = folder2.replace("\\", "/")


def clean_text(x):
    x = x.replace(folder, "").replace(folder2, "")


root, nodes = profile2graph(ps, clean_text=clean_text)
text = root.to_text(fct_width=80)
print(text)
