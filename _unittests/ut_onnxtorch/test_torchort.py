"""
@brief      test log(time=3s)
"""
import logging
import unittest
from pyquickhelper.pycode import ExtTestCase
import numpy
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxRelu, OnnxMatMul)
from mlprodict.onnx_tools.onnx_manipulations import onnx_rename_names
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
import torch
from torch.autograd import Function  # pylint: disable=C0411
from deeponnxcustom import __max_supported_opset__ as TARGET_OPSET
from deeponnxcustom.tools.onnx_helper import onnx_rename_weights
from deeponnxcustom.onnxtorch import TorchOrtFactory


class TestTorchOrt(ExtTestCase):

    class MyReLUAdd(Function):  # pylint: disable=W0223
        @staticmethod
        def forward(ctx, x_input):  # pylint: disable=W0221
            ctx.save_for_backward(x_input)
            return x_input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):  # pylint: disable=W0221
            x_input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[x_input < 0] = 0
            return grad_input

    @staticmethod
    def MyReLUAdd_onnx(N, D_in, H, D_out, rename):
        var = [('X', FloatTensorType([N, D_in]))]
        w1 = numpy.random.randn(D_in, H).astype(numpy.float32)
        w2 = numpy.random.randn(H, D_out).astype(numpy.float32)
        opv = TARGET_OPSET
        onx_alg = OnnxMatMul(
            OnnxRelu(OnnxMatMul(*var, w1, op_version=opv),
                     op_version=opv),
            w2, op_version=opv, output_names=['Y'])
        onx = onx_alg.to_onnx(
            var, target_opset=opv, outputs=[('Y', FloatTensorType())])

        weights = ['Ma_MatMulcst', 'Ma_MatMulcst1']
        if rename:
            names = ['W2', 'W1']
            onx = onnx_rename_names(onx, replace=dict(zip(weights, names)))
            weights = names
        return onx, weights

    @staticmethod
    def _check_(cls, device, dtype, x, y, H, requires_grad):  # pylint: disable=W0211
        # Create random Tensors for weights.
        D_in = x.shape[1]
        D_out = y.shape[1]
        w1 = torch.randn(D_in, H, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=requires_grad)
        w2 = torch.randn(H, D_out, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=requires_grad)

        # simple forward
        if cls == TestTorchOrt.MyReLUAdd:
            with torch.no_grad():
                cls.apply(x.mm(w1)).mm(w2)
        else:
            with torch.no_grad():
                y1 = cls.apply(x, w1, w2)
                y2 = TestTorchOrt.MyReLUAdd.apply(x.mm(w1)).mm(w2)
                diff = (y1 - y2).abs().sum()
                if diff > 1e-4:
                    raise AssertionError(
                        "Discrepancies %r: %r != %r." % (diff, y1, y2))

    @staticmethod
    def _assert_grad_almost_equal(a, b, decimal=4):
        if a.grad is None:
            raise AssertionError("a.grad is None")
        if b.grad is None:
            raise AssertionError("b.grad is None")
        diff = (a.grad - b.grad).abs().sum()
        if diff > 0.1 ** decimal:
            raise AssertionError(
                "Discrepancies %r: %r != %r." % (diff, a.grad, b.grad))

    @staticmethod
    def _check_gradient_(cls, device, dtype, x, y, H,  # pylint: disable=W0211
                         learning_rate=1e-6):
        D_in = x.shape[1]
        D_out = y.shape[1]
        # Create random Tensors for weights.
        w1 = torch.randn(D_in, H, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=True)
        w2 = torch.randn(H, D_out, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=True)
        w1c = torch.clone(  # pylint: disable=E1101
            w1.detach()).requires_grad_(True)
        w2c = torch.clone(  # pylint: disable=E1101
            w2.detach()).requires_grad_(True)

        if cls == TestTorchOrt.MyReLUAdd:
            y_pred = cls.apply(x.mm(w1)).mm(w2)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()
        else:
            y_predc = TestTorchOrt.MyReLUAdd.apply(x.mm(w1c)).mm(w2c)
            lossc = (y_predc - y).pow(2).sum()
            lossc.backward()

            y_pred = cls.apply(x, w1, w2)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()

            TestTorchOrt._assert_grad_almost_equal(w1, w1c)
            TestTorchOrt._assert_grad_almost_equal(w2, w2c)

    def common_gradient(self, rename):
        dtype = torch.float  # pylint: disable=E1101
        device = torch.device("cpu")  # pylint: disable=E1101

        N, D_in, H, D_out = 4, 5, 3, 2

        # Create random Tensors to hold input and outputs.
        x = torch.randn(N, D_in, device=device,  # pylint: disable=E1101
                        dtype=dtype)
        y = torch.randn(N, D_out, device=device,  # pylint: disable=E1101
                        dtype=dtype)

        TestTorchOrt._check_(TestTorchOrt.MyReLUAdd,
                             device, dtype, x, y, H, False)
        TestTorchOrt._check_(TestTorchOrt.MyReLUAdd,
                             device, dtype, x, y, H, True)
        TestTorchOrt._check_gradient_(
            TestTorchOrt.MyReLUAdd, device, dtype, x, y, H)

        onx, weights = TestTorchOrt.MyReLUAdd_onnx(N, D_in, H, D_out, rename)
        if rename:
            self.assertRaise(lambda: TorchOrtFactory(onx, weights), ValueError)
            onx = onnx_rename_weights(onx)
            weights = [init.name for init in onx.graph.initializer]
        fact = TorchOrtFactory(onx, weights)
        self.assertIn('TorchOrtFactory', repr(fact))
        self.assertIn('TorchOrtFactory', str(fact))
        cls = fact.create_class(enable_logging=True, keep_models=True)

        TestTorchOrt._check_(cls, device, dtype, x, y, H, False)
        TestTorchOrt._check_(cls, device, dtype, x, y, H, True)
        TestTorchOrt._check_gradient_(cls, device, dtype, x, y, H)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_gradient(self):
        self.common_gradient(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_gradient_order(self):
        self.common_gradient(True)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_gradient_logging(self):
        logger = logging.getLogger('deeponnxcustom')
        logger.setLevel(logging.DEBUG)
        _, logs = self.assertLogging(
            lambda: self.common_gradient(False), 'deeponnxcustom',
            level=logging.DEBUG)
        self.assertIn("create InferenceSession", logs)
        logger.setLevel(logging.WARNING)

    @staticmethod
    def _check_gradient_iter_(N, cls, device, dtype,  # pylint: disable=W0211
                              x, y, H, learning_rate=1e-6, decimal=4):
        D_in = x.shape[1]
        D_out = y.shape[1]
        # Create random Tensors for weights.
        w1 = torch.randn(D_in, H, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=True)
        w2 = torch.randn(H, D_out, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=True)
        w1c = torch.clone(  # pylint: disable=E1101
            w1.detach()).requires_grad_(True)
        w2c = torch.clone(  # pylint: disable=E1101
            w2.detach()).requires_grad_(True)

        for _ in range(N):
            y_predc = TestTorchOrt.MyReLUAdd.apply(x.mm(w1c)).mm(w2c)
            lossc = (y_predc - y).pow(2).sum()
            lossc.backward()

            y_pred = cls.apply(x, w1, w2)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()

            TestTorchOrt._assert_grad_almost_equal(w1, w1c, decimal=decimal)
            TestTorchOrt._assert_grad_almost_equal(w2, w2c, decimal=decimal)

            with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad
                w1c -= learning_rate * w1c.grad
                w2c -= learning_rate * w2c.grad

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_gradient_one_iteration(self):
        dtype = torch.float  # pylint: disable=E1101
        device = torch.device("cpu")  # pylint: disable=E1101

        # Create random Tensors to hold input and outputs.
        N, D_in, H, D_out = 4, 5, 3, 2
        x = torch.randn(N, D_in, device=device,  # pylint: disable=E1101
                        dtype=dtype)
        y = torch.randn(N, D_out, device=device,  # pylint: disable=E1101
                        dtype=dtype)

        onx, weights = TestTorchOrt.MyReLUAdd_onnx(N, D_in, H, D_out, False)
        fact = TorchOrtFactory(onx, weights)
        cls = fact.create_class(enable_logging=True)
        TestTorchOrt._check_gradient_iter_(
            2, cls, device, dtype, x, y, H, learning_rate=1e-4)
        TestTorchOrt._check_gradient_iter_(
            3, cls, device, dtype, x, y, H, learning_rate=1e-4, decimal=3)
        TestTorchOrt._check_gradient_iter_(
            4, cls, device, dtype, x, y, H, learning_rate=1e-4, decimal=3)

    @staticmethod
    def run_cls(cls, device, dtype, x, y, H,  # pylint: disable=W0211
                learning_rate=1e-6):
        D_in = x.shape[1]
        D_out = y.shape[1]
        # Create random Tensors for weights.
        w1 = torch.randn(D_in, H, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=True)
        w2 = torch.randn(H, D_out, device=device,  # pylint: disable=E1101
                         dtype=dtype, requires_grad=True)

        all_losses = []
        for t in range(500):
            # forward - backward
            if cls == TestTorchOrt.MyReLUAdd:
                y_pred = cls.apply(x.mm(w1)).mm(w2)
                loss = (y_pred - y).pow(2).sum()
                loss.backward()
            else:
                y_pred = cls.apply(x, w1, w2)
                loss = (y_pred - y).pow(2).sum()
                loss.backward()

            # update weights
            with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad
                w1.grad.zero_()
                w2.grad.zero_()
            all_losses.append((t, loss.detach().numpy()))
        return all_losses, w1, w2

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_simple_training(self):
        dtype = torch.float  # pylint: disable=E1101
        device = torch.device("cpu")  # pylint: disable=E1101

        # Create random Tensors to hold input and outputs.
        N, D_in, H, D_out = 4, 5, 3, 2
        x = torch.randn(N, D_in, device=device,  # pylint: disable=E1101
                        dtype=dtype)
        y = torch.randn(N, D_out, device=device,  # pylint: disable=E1101
                        dtype=dtype)

        lr = 1e-5
        all_losses, _, __ = TestTorchOrt.run_cls(
            TestTorchOrt.MyReLUAdd, device, dtype, x, y, H,
            learning_rate=lr)
        # print("TCH", all_losses[0], all_losses[-1], w1.shape,
        #       w1.sum(), w2.shape, w2.sum())
        self.assertGreater(all_losses[0][1], all_losses[-1][1])

        # onnxruntime

        onx, weights = TestTorchOrt.MyReLUAdd_onnx(N, D_in, H, D_out, False)
        fact = TorchOrtFactory(onx, weights)
        cls = fact.create_class(enable_logging=True)

        all_losses2, _, __ = TestTorchOrt.run_cls(
            cls, device, dtype, x, y, H, learning_rate=lr)
        # print("ORT", all_losses2[0], all_losses2[-1], w12.shape,
        #       w12.sum(), w22.shape, w22.sum())
        self.assertGreater(all_losses2[0][1], all_losses2[-1][1])


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('deeponnxcustom')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestTorchOrt().test_gradient_order()
    unittest.main()
