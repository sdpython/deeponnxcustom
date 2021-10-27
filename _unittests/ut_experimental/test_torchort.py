"""
@brief      test log(time=3s)
"""

import unittest
from pyquickhelper.pycode import ExtTestCase
import numpy
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxRelu, OnnxMatMul)
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
import torch
from torch.autograd import Function  # pylint: disable=C0411
from deeponnxcustom.experimental.torchort import TorchOrtFactory


class TestTorchOrt(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_simple(self):

        class MyReLUAdd(Function):
            @staticmethod
            def forward(ctx, x_input):  # pylint: disable=W0221
                ctx.save_for_backward(x_input)
                return x_input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                x_input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

        dtype = torch.float  # pylint: disable=E1101
        device = torch.device("cpu")  # pylint: disable=E1101

        N, D_in, H, D_out = 4, 5, 3, 2

        # Create random Tensors to hold input and outputs.
        x = torch.randn(N, D_in, device=device,  # pylint: disable=E1101
                        dtype=dtype)
        y = torch.randn(N, D_out, device=device,  # pylint: disable=E1101
                        dtype=dtype)

        def run_cls(cls, x, y):
            # Create random Tensors for weights.
            w1 = torch.randn(D_in, H, device=device,  # pylint: disable=E1101
                             dtype=dtype, requires_grad=True)
            w2 = torch.randn(H, D_out, device=device,  # pylint: disable=E1101
                             dtype=dtype, requires_grad=True)

            # simple forward
            with torch.no_grad():
                cls.apply(x.mm(w1)).mm(w2)

            all_losses = []
            learning_rate = 1e-6
            for t in range(500):
                # forward - backward
                y_pred = cls.apply(x.mm(w1)).mm(w2)
                loss = (y_pred - y).pow(2).sum()
                loss.backward()

                # update weights
                with torch.no_grad():
                    w1 -= learning_rate * w1.grad
                    w2 -= learning_rate * w2.grad

                    # Manually zero the gradients after updating weights
                    w1.grad.zero_()
                    w2.grad.zero_()

                all_losses.append((t, loss.detach().numpy()))

            return all_losses, w1, w2

        all_losses, w1, w2 = run_cls(MyReLUAdd, x, y)
        print("Torch", all_losses[-1], w1.shape, w1.sum(), w2.shape, w2.sum())

        var = [('X', FloatTensorType([N, D_in]))]
        w1 = numpy.random.randn(D_in, H).astype(numpy.float32)
        w2 = numpy.random.randn(H, D_out).astype(numpy.float32)
        opv = 14
        onx_alg = OnnxMatMul(
            OnnxRelu(OnnxMatMul(*var, w1, op_version=opv),
                     op_version=opv),
            w2, op_version=opv, output_names=['Y'])
        onx = onx_alg.to_onnx(
            var, target_opset=opv, outputs=[('Y', FloatTensorType())])
        with open("model_ooo.onnx", "wb") as f:
            f.write(onx.SerializeToString())

        weights = ['Ma_MatMulcst', 'Ma_MatMulcst1']
        fact = TorchOrtFactory(onx, weights)
        cls = fact.create_class(enable_logging=True)

        all_losses2, w12, w22 = run_cls(cls, x, y)
        print("Torch", all_losses2[-1], w12.shape,
              w12.sum(), w22.shape, w22.sum())


if __name__ == "__main__":
    import logging
    logger = logging.getLogger('deeponnxcustom')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    unittest.main()
