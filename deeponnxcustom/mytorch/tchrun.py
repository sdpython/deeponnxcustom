"""
@file
@brief Executes ONNX graph with pytorch.
"""
from onnx.numpy_helper import to_array
import torch
from ..tools.math_helper import decompose_permutation


class _function_OnnxTorchRuntime:

    @staticmethod
    def _concat(*tensors, axis=0):
        nonnull = [t for t in tensors if len(t.shape) > 0]
        if len(nonnull) == 0:
            raise NotImplementedError(
                "Cannot concatenate empty tensors.")
        if len(nonnull) == 1:
            return nonnull[0]
        try:
            return torch.cat(nonnull, dim=axis)  # pylint: disable=E1101
        except RuntimeError as e:
            raise RuntimeError(
                "Unable to run 'cat' with shape=%r and axis=%r." % (
                    ", ".join(str(t.shape) for t in tensors),
                    axis)) from e

    @staticmethod
    def _gather(t, indices, axis=0):
        return torch.gather(t, axis, indices)  # pylint: disable=E1101

    @staticmethod
    def _gemm(a, b, c=None, alpha=1, beta=0, transA=False, transB=False):
        if transA:
            a = a.T
        if transB:
            b = b.T
        res = torch.matmul(a, b) * alpha  # pylint: disable=E1101
        if c is not None:
            res += c * beta
        return res

    @staticmethod
    def _reduceprod(data, axes=None, keepdims=1):
        if axes is None:
            if len(data.shape) == 1:
                return torch.prod(  # pylint: disable=E1101
                    data, 0, keepdims == 1)
            raise NotImplementedError(
                "Unable to prod(...) with shape=%r axes=%r keepdims=%r." % (
                    tuple(data.shape), axes, keepdims))
        return torch.prod(  # pylint: disable=E1101
            data, dim=axes, keepdim=keepdims == 1)

    @staticmethod
    def _reducesum(data, axes=None, keepdims=1):
        if axes is None:
            if len(data.shape) == 1:
                return torch.sum(  # pylint: disable=E1101
                    data, 0, keepdims == 1)
            raise NotImplementedError(
                "Unable to prod(...) with shape=%r axes=%r keepdims=%r." % (
                    tuple(data.shape), axes, keepdims))
        return torch.sum(  # pylint: disable=E1101
            data, dim=axes, keepdim=keepdims == 1)

    @staticmethod
    def _reshape(t, shape):
        return torch.reshape(t, tuple(shape))  # pylint: disable=E1101

    @staticmethod
    def _shape(t):
        return torch.tensor(t.shape)  # pylint: disable=E1101

    @staticmethod
    def _squeeze(data, axes=None):
        if axes is None:
            return torch.squeeze(data)  # pylint: disable=E1101
        if len(axes) == 1:
            return torch.squeeze(data, axes[0])  # pylint: disable=E1101
        for a in reversed(axes):
            data = torch.squeeze(data, a)  # pylint: disable=E1101
        return data

    @staticmethod
    def _transpose(t, perm):
        transitions = decompose_permutation(perm)
        for a, b in transitions:
            t = torch.transpose(t, a, b)  # pylint: disable=E1101
        return t

    @staticmethod
    def _unqueeze(t, dim):
        if tuple(dim.shape) == (0, ):
            return t
        if len(dim) == 1:
            return torch.unsqueeze(t, dim[0])  # pylint: disable=E1101
        v = t
        for d in dim:
            v = torch.unsqueeze(v, d)  # pylint: disable=E1101
        return v


class OnnxTorchRuntime:
    """
    Executes ONNX graph using :epkg:`torch` function.
    """

    _mapping = {
        'Concat': _function_OnnxTorchRuntime._concat,
        'Gather': _function_OnnxTorchRuntime._gather,
        'Gemm': _function_OnnxTorchRuntime._gemm,
        'Identity': lambda x: x,
        'MatMul': torch.matmul,  # pylint: disable=E1101
        'Max': torch.max,  # pylint: disable=E1101
        'ReduceProd':
            _function_OnnxTorchRuntime._reduceprod,  # pylint: disable=E1101
        'ReduceSum':
            _function_OnnxTorchRuntime._reducesum,  # pylint: disable=E1101
        'Reshape': _function_OnnxTorchRuntime._reshape,
        'Shape': _function_OnnxTorchRuntime._shape,
        'Squeeze': _function_OnnxTorchRuntime._squeeze,
        'Transpose': _function_OnnxTorchRuntime._transpose,
        'Unsqueeze': _function_OnnxTorchRuntime._unqueeze,
    }

    def __init__(self, onnx_model):
        self._onnx_model = onnx_model
        self._inits = OnnxTorchRuntime._extract_init(onnx_model)
        self._atts = OnnxTorchRuntime._extract_atts(onnx_model)

    @staticmethod
    def _extract_init(onnx_model):
        """
        Builds a dictionary with all initializers
        converted into torch arrays.
        """
        res = {}
        for init in onnx_model.graph.initializer:
            if init.name in res:
                raise RuntimeError(
                    "Duplicated initializer name %r for type %r." % (
                        init.name, init.op_type))
            res[init.name] = torch.from_numpy(  # pylint: disable=E1101
                to_array(init))
        return res

    @staticmethod
    def _extract_atts(onnx_model):
        """
        Builds a dictionary with all attributes
        """
        res = {}
        for i, node in enumerate(onnx_model.graph.node):
            node_name = "N%d_%s" % (i, node.name)
            res[node_name] = {}
            for at in node.attribute:
                if node.op_type in ('ReduceSum', 'ReduceProd'):
                    if at.name == 'axes':
                        res[node_name][at.name] = tuple(at.ints)
                    else:
                        res[node_name][at.name] = at.i
                if node.op_type == 'Transpose':
                    res[node_name][at.name] = tuple(at.ints)
                elif node.op_type == 'Gather':
                    res[node_name][at.name] = at.i
                elif node.op_type == 'Gemm':
                    if at.name in ('alpha', 'beta'):
                        res[node_name][at.name] = at.f
                    else:
                        res[node_name][at.name] = at.i
        return res

    def _run_op(self, node_name, node, *inputs):
        """
        Executes a node with :epkg:`pytorch`.
        Returns a dictionary.
        """
        if len(node.output) != 1:
            raise NotImplementedError(
                "Unable to execute a node with more than one "
                "input (type=%r)." % node.op_type)
        tf = OnnxTorchRuntime._mapping[node.op_type]
        try:
            res = tf(*inputs, **self._atts[node_name])
        except (TypeError, IndexError, RuntimeError) as e:
            raise RuntimeError(
                "Unable to run operator %r with len(inputs)=%d, atts=%r.\n%r"
                "" % (node.op_type, len(inputs),
                      self._atts[node_name], inputs)) from e
        if isinstance(res, tuple):
            return res
        return (res, )

    def run(self, *inputs, verbose=False):
        """
        Executes the ONNX graph.

        :param inputs: inputs of the function
        :param verbose: displays more information while running the graph
        :return: a result or a tuple of results
        """
        keep = self._inits.copy()
        for i, v in zip(self._onnx_model.graph.input, inputs):
            keep[i.name] = v

        for i, node in enumerate(self._onnx_model.graph.node):
            node_name = "N%d_%s" % (i, node.name)
            node_inputs = [keep[name] for name in node.input]
            res = self._run_op(node_name, node, *node_inputs)
            if verbose:
                print(  # pragma: no cover
                    "[OnnxTorchRuntime.run] op=%r, shapes=[%s] "
                    "-> %s, name=%r in [%r, %r], atts=%r" % (
                        node.op_type,
                        ", ".join(map(
                            lambda x: str(tuple(getattr(x, 'shape', '?'))),
                            node_inputs)),
                        ", ".join(map(
                            lambda x: str(tuple(getattr(x, 'shape', '?'))),
                            res)),
                        node.name,
                        float(min(t.min() for t in res)),
                        float(max(t.max() for t in res)),
                        self._atts[node_name]))
            for name, value in zip(node.output, res):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "Unexpected value for name=%r, type=%r." % (
                            name, type(value)))
                keep[name] = value

        res = tuple(keep[o.name] for o in self._onnx_model.graph.output)
        if len(res) == 1:
            return res[0]
        return res
