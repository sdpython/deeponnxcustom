"""
.. _l-bench-torch-ort-factory:

Profling functions while using ONNX to extend pytorch
=====================================================

The example creates a simple graph with many inputs so that
the graph computing the gradient has many outputs.
As the training of the whole model is done by :epkg:`torch`,
some time is spent just to exchange information between :epkg:`torch`
and :epkg:`onnxruntime`. This time is minimized because the data is
exchanged through :epkg:`DLPack` protocol. That leaves the
copy of the structures describing the data.

.. contents::
    :local:

ONNX graph
++++++++++

"""
import os
import time
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxSigmoid, OnnxMatMul, OnnxAdd)
from pyquickhelper.pycode.profiling import profile, profile2graph, profile2df
from mlprodict.onnx_tools.onnx_manipulations import onnx_rename_names
from mlprodict.plotting.plotting_onnx import plot_onnx
import torch
from deeponnxcustom.onnxtorch.torchort import TorchOrtFactory


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
onx, weights = create_onnx_graph(N, n_loops=20)

plot_onnx(onx.SerializeToString(), temp_dot="plot_profile_torchort.dot")


###############################################
# Wraps ONNX as a torch.autograd.Function
# +++++++++++++++++++++++++++++++++++++++
#
# Let's build a torch function with class
# :class:`TorchOrtFactory
# <deeponnxcustom.onnxtorch.torchort.TorchOrtFactory>`.

fact = TorchOrtFactory(onx, [w[0] for w in weights])
cls = fact.create_class(keep_models=True)
print("torch version:", torch.__version__)
print(cls)

##########################################
# The gradient graph looks like this:

fix, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_onnx(cls._trained_onnx, ax=ax)

##########################################
# Training
# ++++++++
#
# The training happens on cpu or gpu depending on what is
# available. We try first a few iteation to see how it goes.


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
train_losses, final_weights = train_cls(cls, device, x, y, weights, n_iter=10)
train_losses = numpy.array([t.cpu().detach().numpy().ravel()
                            for t in train_losses])
df = DataFrame(data=train_losses, columns=['iter', 'train_loss'])
df.plot(x="iter", y="train_loss", title="Training loss")

#######################################
# Profiling
# +++++++++
#
# We run many more iterations and profile the execution.

folder = os.path.abspath(os.getcwd()).split('deeponnxcustom')[0]
folder2 = os.path.abspath(os.path.split(
    os.path.dirname(torch.__file__))[0])[:-6]

# Same class but without any unnecessary data.
cls = fact.create_class()

begin = time.perf_counter()
train_cls(cls, device, x, y, weights, n_iter=200)
print("total time: %r" % (time.perf_counter() - begin))

########################################
# Full profile as text.

ps, text = profile(
    lambda: train_cls(cls, device, x, y, weights, n_iter=200))
print(type(ps))
print(text.replace(folder, "").replace(folder2, ""))

########################################
# Same results in a graph.


df = profile2df(ps)
ax = df[['fct', 'cum_tall']].head(n=15).set_index(
    'fct').plot(kind='bar', figsize=(8, 3), rot=30)
ax.set_title("example of a graph")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right')

########################################
# Presentation with partial call stack
# ++++++++++++++++++++++++++++++++++++
#
# The previous presentation do not show any information
# about where a function is called from. Let's use
# function :func:`profile2graph
# <pyquickhelper.pycode.profiling.profile2graph>`.

folder = folder.replace("\\", "/")
folder2 = folder2.replace("\\", "/")


def clean_text(x):
    x = x.replace(folder, "").replace(folder2, "")


root, nodes = profile2graph(ps, clean_text=clean_text)
text = root.to_text(fct_width=70)
print(text)


# plt.show()
