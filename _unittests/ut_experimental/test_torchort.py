"""
@brief      test log(time=3s)
"""

import unittest
from pyquickhelper.pycode import ExtTestCase
import numpy
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxRelu
from deeponnxcustom.experimental.torchort import TorchOrtFactory
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
import torch
from torch.autograd import Function


class TestTorchOrt(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_simple(self):
        
        class MyReLU(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input        

        dtype = torch.float
        device = torch.device("cpu")

        N, D_in, H, D_out = 64, 1000, 100, 10

        # Create random Tensors to hold input and outputs.
        x = torch.randn(N, D_in, device=device, dtype=dtype)
        y = torch.randn(N, D_out, device=device, dtype=dtype)

        def run_cls(cls, x, y):
            # Create random Tensors for weights.
            w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
            w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
            
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
        
        all_losses, w1, w2 = run_cls(MyReLU, x, y)
        print("Torch", all_losses[-1], w1.shape, w1.sum(), w2.shape, w2.sum())
        
        var = [('X', FloatTensorType())]
        onx = OnnxRelu(*var, op_version=14, output_names=['Y']).to_onnx(
            var, target_opset=14, outputs=[('Y', FloatTensorType())])
        weights = ['X']
        
        fact = TorchOrtFactory(onx, weights)
        cls = fact.create_class(enable_logging=True)
        
        all_losses2, w12, w22 = run_cls(cls, x, y)
        print("Torch", all_losses2[-1], w12.shape, w12.sum(), w22.shape, w22.sum())




if __name__ == "__main__":
    import logging
    logger = logging.getLogger('deeponnxcustom')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()
