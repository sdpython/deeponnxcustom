"""
.. _l-einsum:

Compares implementations of Einsum
==================================

This example compares the performance of :func:`numpy.einsum`,
:func:`torch.einsum` and its decomposition into standard
vector operations for a couple of equations.

.. contents::
    :local:

Available optimisation
++++++++++++++++++++++

The code shows which optimisation is used for the custom
implementation, *AVX* or *SSE* and the number of available processors,
equal to the default number of used threads to parallelize.
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxEinsum
from mlprodict.tools import measure_time
from tqdm import tqdm
from mlprodict.testing.experimental_c import (
    code_optimisation)
from mlprodict.testing.einsum.einsum_fct import _einsum
from mlprodict.plotting.plotting_onnx import plot_onnx
from deeponnxcustom.onnxtorch.tchrun import OnnxTorchRuntime
print(code_optimisation())

###################################
# Einsum: common code
# +++++++++++++++++++
#
# The main function which benchmark a couple of options.
#
# * :func:`numpy.einsum`
# * :func:`torch.einsum`
# * function *einsum* from :epkg:`onnxruntime`
# * decomposition of einsum into ONNX and processed with :epkg:`onnxruntime`

try:
    from torch import einsum as torch_einsum, from_numpy
except ImportError:
    torch_einsum = None


def build_ort_einsum(equation, op_version=14):  # opset=13, 14, ...
    node = OnnxEinsum('x', 'y', equation=equation,
                      op_version=op_version,
                      output_names=['z'])
    onx = node.to_onnx(inputs=[('x', FloatTensorType()),
                               ('y', FloatTensorType())],
                       target_opset=op_version)
    sess = InferenceSession(onx.SerializeToString())
    return lambda x, y: sess.run(None, {'x': x, 'y': y})


def build_ort_decomposed(equation, op_version=14):  # opset=13, 14, ...
    cache = _einsum(equation, numpy.float32, opset=op_version,
                    optimize=True, verbose=True, runtime="python")
    if not hasattr(cache, 'onnx_'):
        cache.build()
    sess = InferenceSession(cache.onnx_.SerializeToString())
    return cache.onnx_, lambda x, y: sess.run(None, {'X0': x, 'X1': y})


def build_torch_decomposed(equation, op_version=14):  # opset=13, 14, ...
    cache = _einsum(equation, numpy.float32, opset=op_version,
                    optimize=True, verbose=True, runtime="python")
    if not hasattr(cache, 'onnx_'):
        cache.build()
    sess = OnnxTorchRuntime(cache.onnx_)
    return cache.onnx_, lambda x, y: sess.run(x, y)


def loop_einsum_eq(fct, equation, xs, ys):
    for x, y in zip(xs, ys):
        fct(equation, x, y)


def loop_einsum_eq_th(fct, equation, xs, ys):
    for x, y in zip(xs, ys):
        fct(equation, x, y, nthread=-1)


def loop_einsum(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


def benchmark_equation(equation, number=5, repeat=3):
    # equations
    ort_einsum = build_ort_einsum(equation)
    einsum_onnx, ort_einsum_decomposed = build_ort_decomposed(equation)
    torch_onnx, ort_torch_decomposed = build_torch_decomposed(equation)

    K, S, M, E = 16, 1024, 768, 64
    C = S // E * 2
    SIZE_MAP = {'K': K, 'S': S, 'E': E, 'C': C, 'M': M}

    pos1 = equation.find(',')
    pos2 = equation.find('->')
    lhs_op = equation[0:pos1]
    rhs_op = equation[pos1 + 1:pos2]
    lhs_shape = []
    for c in lhs_op:
        lhs_shape.append(SIZE_MAP[c.upper()])
    rhs_shape = []
    for c in rhs_op:
        rhs_shape.append(SIZE_MAP[c.upper()])

    terms = equation.split('->')[0].split(',')
    if 'e' in equation:
        pos_left = terms[0].find('e')
        pos_right = terms[1].find('e')
    else:
        pos_left = terms[0].find('k')
        pos_right = terms[1].find('k')

    def left_dim(dim):
        if pos_left == -1:
            return lhs_shape
        cp = list(lhs_shape)
        cp[pos_left] = dim
        return tuple(cp)

    def right_dim(dim):
        if pos_right == -1:
            return rhs_shape
        cp = list(rhs_shape)
        cp[pos_right] = dim
        return tuple(cp)

    sizes = [8, 16, 32, 64, 128, 256]
    if max(len(rhs_shape), len(lhs_shape)) >= 3:
        sizes = sizes[:4]

    res = []
    for dim in tqdm(sizes):
        xs = [numpy.random.rand(*left_dim(dim)).astype(numpy.float32)
              for _ in range(5)]
        ys = [numpy.random.rand(*right_dim(dim)).astype(numpy.float32)
              for _ in range(5)]

        # numpy
        ctx = dict(equation=equation, xs=xs, ys=ys, einsum=numpy.einsum,
                   loop_einsum=loop_einsum, loop_einsum_eq=loop_einsum_eq,
                   loop_einsum_eq_th=loop_einsum_eq_th)
        obs = measure_time(
            "loop_einsum_eq(einsum, equation, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'numpy.einsum'
        res.append(obs)

        # onnxruntime
        ctx['einsum'] = ort_einsum
        obs = measure_time(
            "loop_einsum(einsum, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort_einsum'
        res.append(obs)

        # onnxruntime decomposed
        ctx['einsum'] = ort_einsum_decomposed
        obs = measure_time(
            "loop_einsum(einsum, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'ort_dec'
        res.append(obs)

        # torch decomposed
        ctx['einsum'] = ort_torch_decomposed
        ctx['xs'] = [from_numpy(x) for x in xs]
        ctx['ys'] = [from_numpy(y) for y in ys]
        obs = measure_time(
            "loop_einsum(einsum, xs, ys)",
            div_by_number=True, context=ctx, repeat=repeat, number=number)
        obs['dim'] = dim
        obs['fct'] = 'torch_dec'
        res.append(obs)

        if torch_einsum is not None:
            # torch
            ctx['einsum'] = torch_einsum
            obs = measure_time(
                "loop_einsum_eq(einsum, equation, xs, ys)",
                div_by_number=True, context=ctx, repeat=repeat, number=number)
            obs['dim'] = dim
            obs['fct'] = 'torch_einsum'
            res.append(obs)

    # Dataframes
    df = pandas.DataFrame(res)
    piv = df.pivot('dim', 'fct', 'average')

    rs = piv.copy()
    rs['ort_einsum'] = rs['numpy.einsum'] / rs['ort_einsum']
    rs['ort_dec'] = rs['numpy.einsum'] / rs['ort_dec']
    if 'torch_einsum' in rs.columns:
        rs['torch_einsum'] = rs['numpy.einsum'] / rs['torch_einsum']
    rs['numpy.einsum'] = 1.

    # Graphs.
    shapes = ("%s - %s" % (left_dim('N'), right_dim('N'))).replace("'", "")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    piv.plot(logx=True, logy=True, ax=ax[0],
             title="Einsum benchmark\n%s -- %s"
                   " lower better" % (shapes, equation))
    ax[0].legend(prop={"size": 9})
    rs.plot(logx=True, logy=True, ax=ax[1],
            title="Einsum Speedup, baseline=numpy\n%s -- %s"
                  " higher better" % (shapes, equation))
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], 'g--')
    ax[1].plot([min(rs.index), max(rs.index)], [2., 2.], 'g--')
    ax[1].legend(prop={"size": 9})

    return df, rs, ax, einsum_onnx

#################################
# A last function to plot the ONNX graphs.


def plot_onnx_einsum(equation, onx):
    filename = "einsum_eq_%s.onnx" % (
        equation.replace(",", "_").replace("->", "__"))
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = plot_onnx(onx, ax=ax)
    ax.set_title(equation)
    return ax

###################################
# First equation: s,se->se
# ++++++++++++++++++++++++
#


dfs = []
equation = "s,se->se"
df, piv, ax, onx = benchmark_equation(equation)
df.pivot("fct", "dim", "average")
dfs.append(df)
df.pivot("dim", "fct", "average")

####################################
# The onnx decomposition.

plot_onnx_einsum(equation, onx)

###################################
# Second equation: se,sc->sec
# +++++++++++++++++++++++++++
#

dfs = []
equation = "se,sc->sec"
df, piv, ax, onx = benchmark_equation(equation)
df.pivot("fct", "dim", "average")
dfs.append(df)
df.pivot("dim", "fct", "average")

####################################
# The onnx decomposition.

plot_onnx_einsum(equation, onx)

###################################
# Third equation: se,se->s
# ++++++++++++++++++++++++
#

dfs = []
equation = "se,se->s"
df, piv, ax, onx = benchmark_equation(equation)
df.pivot("fct", "dim", "average")
dfs.append(df)
df.pivot("dim", "fct", "average")

####################################
# The onnx decomposition.

plot_onnx_einsum(equation, onx)

###################################
# Fourth equation: ks,ksm->sm
# ++++++++++++++++++++++++++
#

dfs = []
equation = "ks,ksm->sm"
df, piv, ax, onx = benchmark_equation(equation)
df.pivot("fct", "dim", "average")
dfs.append(df)
df.pivot("dim", "fct", "average")

####################################
# The onnx decomposition.

plot_onnx_einsum(equation, onx)

###################################
# Fifth equation: sec,sm->ecm
# +++++++++++++++++++++++++++
#

dfs = []
equation = "sec,sm->ecm"
df, piv, ax, onx = benchmark_equation(equation, number=1, repeat=1)
df.pivot("fct", "dim", "average")
dfs.append(df)
df.pivot("dim", "fct", "average")

####################################
# The onnx decomposition.

plot_onnx_einsum(equation, onx)

###################################
# Sixth equation: sec,ecm->sm
# +++++++++++++++++++++++++++
#

dfs = []
equation = "sec,ecm->sm"
df, piv, ax, onx = benchmark_equation(equation, number=1, repeat=1)
df.pivot("fct", "dim", "average")
dfs.append(df)
df.pivot("dim", "fct", "average")

####################################
# The onnx decomposition.

plot_onnx_einsum(equation, onx)

####################################
# Conclusion
# ++++++++++
#
# pytorch seems quite efficient on these examples.

merged = pandas.concat(dfs)
name = "einsum"
merged.to_csv("plot_%s.csv" % name, index=False)
merged.to_excel("plot_%s.xlsx" % name, index=False)
plt.savefig("plot_%s.png" % name)

# plt.show()
