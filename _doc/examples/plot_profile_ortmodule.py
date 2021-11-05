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
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from pyquickhelper.pycode.profiling import profile, profile2graph, profile2df
import torch
from onnxruntime.training.ortmodule import ORTModule


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


class NLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out, lay=2):
        super(NLayerNet, self).__init__()
        H = 2
        self.linears = [torch.nn.Linear(D_in, H)
                        for n in range(lay)]
        self.linear2 = torch.nn.Linear(H * lay, D_out)

    def forward(self, x):
        xs = [torch.sigmoid((x)) for lay in self.linears]
        conc = torch.cat(xs, dim=1)
        y_pred = self.linear2(conc)
        return y_pred


##########################################
# Training
# ++++++++
#
# The training happens on cpu or gpu depending on what is
# available. We try first a few iteation to see how it goes.


def train_model(model, device, x, y, n_iter=100, learning_rate=1e-4):
    model = model.to(device)
    x = from_numpy(x, requires_grad=True, device=device)
    y = from_numpy(y, requires_grad=True, device=device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for t in range(n_iter):

        def step_train():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss

        loss = step_train()
        losses.append(loss)

    return losses


device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("device:", device)

d_in, d_out, N = 2, 1, 100
x = numpy.random.randn(N, d_in).astype(numpy.float32)
y = numpy.random.randn(N, d_out).astype(numpy.float32)
model = ORTModule(NLayerNet(d_in, d_out))

train_losses = train_model(model, device, x, y, n_iter=10)
train_losses = numpy.array([t.detach().numpy().ravel() for t in train_losses])

df = DataFrame(data=train_losses, columns=['train_loss'])
df['iter'] = df.index + 1
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df.plot(x="iter", y="train_loss", title="Training loss", ax=ax)

#######################################
# Profiling
# +++++++++
#
########################################
# Full profile as text.

folder = os.path.abspath(os.getcwd()).split('deeponnxcustom')[0]
folder2 = os.path.abspath(os.path.split(
    os.path.dirname(torch.__file__))[0])[:-6]

ps, text = profile(
    lambda: train_model(model, device, x, y, n_iter=200))
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
