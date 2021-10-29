"""
@file
@brief Executes ONNX graph with pytorch.
"""
from onnx.numpy_helper import to_array
import torch


class _function_OnnxTorchRuntime:

    @staticmethod
    def _concat(*tensors, axis=0):
        return torch.cat(tensors, dim=axis)  # pylint: disable=E1101

    @staticmethod
    def _gather(t, indices, axis=0):
        return torch.gather(t, axis, indices)  # pylint: disable=E1101

    @staticmethod
    def _gemm(a, b, c=None, alpha=1, beta=0, transA=False, transB=False):
        if transA:
            a = a.T
        if transB:
            b = b.T
        res = a @ b * alpha
        if c is not None:
            res += c * beta
        return res

    @staticmethod
    def _reshape(t, shape):
        return torch.reshape(t, tuple(shape))  # pylint: disable=E1101

    @staticmethod
    def _shape(t):
        return torch.tensor(t.shape)  # pylint: disable=E1101

    @staticmethod
    def _transpose(t, perm):
        swapped = []
        for i, p in enumerate(perm):
            if i != p:
                swapped.append(p)
        if len(swapped) == 2:
            return torch.transpose(t, *swapped)  # pylint: disable=E1101
        if perm == (1, 2, 0):
            t1 = torch.transpose(t, 2, 0)  # pylint: disable=E1101
            return torch.transpose(t1, 1, 0)  # pylint: disable=E1101
        if perm == (2, 0, 1):
            t1 = torch.transpose(t, 1, 0)  # pylint: disable=E1101
            return torch.transpose(t1, 2, 0)  # pylint: disable=E1101
        raise NotImplementedError(
            "Unable to permute more than two axes %r and shape=%r."
            "" % (perm, t.shape))

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
        'ReduceSum': torch.sum,  # pylint: disable=E1101
        'Reshape': _function_OnnxTorchRuntime._reshape,
        'Shape': _function_OnnxTorchRuntime._shape,
        'Squeeze': lambda t, dim: torch.squeeze(  # pylint: disable=E1101
            t, dim[0]),
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
            res[init.name] = torch.from_numpy(  # pylint: disable=E1101
                to_array(init))
        return res

    @staticmethod
    def _extract_atts(onnx_model):
        """
        Builds a dictionary with all attributes
        """
        res = {}
        for node in onnx_model.graph.node:
            res[node.name] = {}
            for at in node.attribute:
                if node.op_type == 'Transpose':
                    res[node.name][at.name] = tuple(at.ints)
                elif node.op_type == 'Gather':
                    res[node.name][at.name] = at.i
                elif node.op_type == 'Gemm':
                    if at.name in ('alpha', 'beta'):
                        res[node.name][at.name] = at.f
                    else:
                        res[node.name][at.name] = at.i
        return res

    def _run_op(self, node, *inputs):
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
            res = tf(*inputs, **self._atts[node.name])
        except (TypeError, IndexError, RuntimeError) as e:
            raise RuntimeError(
                "Unable to run operator %r with len(inputs)=%d, atts=%r.\n%r"
                "" % (node.op_type, len(inputs),
                      self._atts[node.name], inputs)) from e
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

        for node in self._onnx_model.graph.node:
            inputs = [keep[name] for name in node.input]
            if verbose:
                print("[OnnxTorchRuntime.run] op=%r, name=%r, shapes=[%r], "
                      "atts=%r" % (
                          node.op_type, node.name,
                          ", ".join(map(lambda x: str(getattr(x, 'shape', '?')),
                                        inputs)),
                          self._atts[node.name]))
            res = self._run_op(node, *inputs)
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
