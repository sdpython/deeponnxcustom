"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxConcat, OnnxGemm, OnnxMatMul,
    OnnxReduceProd, OnnxReduceSum_11, OnnxSqueeze)
from mlprodict.testing.einsum.einsum_fct import _einsum
from mlprodict.onnx_tools.onnx_manipulations import onnx_rename_names
import torch
from torch import from_numpy  # pylint: disable=E0611
from deeponnxcustom.onnxtorch.tchrun import OnnxTorchRuntime


class TestOnnxTorchRuntime(ExtTestCase):

    _opv = 14

    def common_test(self, onnx_model, expected, *inputs, verbose=False):
        rt = OnnxTorchRuntime(onnx_model)
        res = rt.run(*inputs, verbose=verbose)
        self.assertEqualArray(expected.numpy(), res.numpy(), decimal=5)

    @ignore_warnings(UserWarning)
    def test_onnx_torch_runtime(self):
        cst = numpy.array([[0, 1], [2, 3]], dtype=numpy.float32)
        node = OnnxMatMul('X', cst, op_version=TestOnnxTorchRuntime._opv,
                          output_names=['Y'])
        onx = node.to_onnx([('X', FloatTensorType())],
                           target_opset=TestOnnxTorchRuntime._opv)

        tx = torch.randn(2, 2)  # pylint: disable=E1101
        expected = tx @ torch.from_numpy(cst)  # pylint: disable=E1101
        self.common_test(onx, expected, tx)

    @ignore_warnings(UserWarning)
    def test_einsum(self):
        equations = ['ks,ksm->sm', 's,se->se', 'se,sc->sec',
                     'se,se->s']

        for eq in equations:
            with self.subTest(eq=eq):
                cache = _einsum(
                    eq, numpy.float32, opset=TestOnnxTorchRuntime._opv,
                    optimize=False, verbose=False, runtime="python")
                onx = cache.onnx_
                terms = eq.split('->', maxsplit=1)[0].split(',')
                shape1 = (3, ) * len(terms[0])
                shape2 = (3, ) * len(terms[1])
                tx1 = torch.randn(*shape1)  # pylint: disable=E1101
                tx2 = torch.randn(*shape2)  # pylint: disable=E1101
                expected = torch.einsum(eq, tx1, tx2)
                self.common_test(onx, expected, tx1, tx2, verbose=False)

    @ignore_warnings(UserWarning)
    def test_einsum2(self):
        equations = ['sec,ecm->sm', 'sec,sm->ecm']

        for eq in equations:
            with self.subTest(eq=eq):
                cache = _einsum(
                    eq, numpy.float32, opset=TestOnnxTorchRuntime._opv,
                    optimize=False, verbose=False, runtime="python")
                onx = cache.onnx_
                onx = onnx_rename_names(onx)
                with open("debug.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                terms = eq.split('->', maxsplit=1)[0].split(',')
                shape1 = (3, ) * len(terms[0])
                shape2 = (3, ) * len(terms[1])
                tx1 = torch.randn(*shape1)  # pylint: disable=E1101
                tx2 = torch.randn(*shape2)  # pylint: disable=E1101
                expected = torch.einsum(eq, tx1, tx2)
                self.common_test(onx, expected, tx1, tx2, verbose=False)

    @ignore_warnings(UserWarning)
    def test_torch_runtime_concatv(self):
        onx = OnnxConcat('X', 'Y', output_names=['Z'],
                         op_version=TestOnnxTorchRuntime._opv)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X), from_numpy(Y))
        self.assertEqualArray(numpy.vstack([X, Y]), Z.numpy())

    @ignore_warnings(UserWarning)
    def test_torch_runtime_concath(self):
        onx = OnnxConcat('X', 'Y', output_names=['Z'],
                         op_version=TestOnnxTorchRuntime._opv,
                         axis=1)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X), from_numpy(Y))
        self.assertEqualArray(numpy.hstack([X, Y]), Z.numpy())

    @ignore_warnings(UserWarning)
    def test_torch_runtime_concat1(self):
        onx = OnnxConcat('X', output_names=['Z'],
                         op_version=TestOnnxTorchRuntime._opv)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(X, Z.numpy())

    def test_onnxt_runtime_gemm(self):
        idi = numpy.array([[1, 0], [1, 1]], dtype=numpy.float32)
        cst = numpy.array([4, 5], dtype=numpy.float32)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)

        onx = OnnxGemm('X', idi, cst, output_names=['Y'],
                       op_version=TestOnnxTorchRuntime._opv,
                       alpha=5., beta=3.)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        exp_Z = numpy.dot(X, idi) * 5 + cst * 3
        self.assertEqualArray(exp_Z, Z.numpy(), decimal=5)

        onx = OnnxGemm('X', idi, cst, output_names=['Y'],
                       op_version=TestOnnxTorchRuntime._opv,
                       alpha=5., beta=3., transA=1)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        exp_Z = numpy.dot(X.T, idi) * 5 + cst * 3
        self.assertEqualArray(exp_Z, Z.numpy(), decimal=5)

    def test_onnxt_runtime_reduce_sum(self):
        cl = OnnxReduceSum_11

        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = cl('X', output_names=['Y'], keepdims=0,
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        self.assertRaise(lambda: tonx.run(from_numpy(X)), RuntimeError)

        onx = cl('X', output_names=['Y'], keepdims=0, axes=[1],
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(X.sum(keepdims=0, axis=1), Z.numpy(), decimal=5)

        onx = cl('X', output_names=['Y'], keepdims=1, axes=[0, 1],
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(
            X.sum(keepdims=1, axis=(0, 1)), Z.numpy(), decimal=5)

        X = X.ravel()
        onx = cl('X', output_names=['Y'], keepdims=0,
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(X.sum(keepdims=0), Z.numpy(), decimal=5)

    def test_onnxt_runtime_reduce_prod(self):
        cl = OnnxReduceProd

        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = cl('X', output_names=['Y'], keepdims=0,
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        self.assertRaise(lambda: tonx.run(from_numpy(X)), RuntimeError)

        onx = cl('X', output_names=['Y'], keepdims=0, axes=[1],
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(X.prod(keepdims=0, axis=1), Z.numpy(), decimal=5)

        onx = cl('X', output_names=['Y'], keepdims=1, axes=[0, 1],
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(
            X.prod(keepdims=1, axis=(0, 1)), Z.numpy(), decimal=5)

        X = X.ravel()
        onx = cl('X', output_names=['Y'], keepdims=0,
                 op_version=TestOnnxTorchRuntime._opv)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(X.prod(keepdims=0), Z.numpy(), decimal=5)

    @ignore_warnings(UserWarning)
    def test_torch_runtime_squeeze(self):
        onx = OnnxSqueeze('X', output_names=['Z'],
                          op_version=TestOnnxTorchRuntime._opv)
        X = numpy.array([[[1, 2], [3, 4]]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=TestOnnxTorchRuntime._opv)
        tonx = OnnxTorchRuntime(model_def)
        Z = tonx.run(from_numpy(X))
        self.assertEqualArray(numpy.squeeze(X), Z.numpy())


if __name__ == "__main__":
    unittest.main()
