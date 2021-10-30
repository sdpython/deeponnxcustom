"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxMatMul)
from mlprodict.testing.einsum.einsum_fct import _einsum
from mlprodict.onnx_tools.onnx_manipulations import onnx_rename_names
import torch
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


if __name__ == "__main__":
    unittest.main()
