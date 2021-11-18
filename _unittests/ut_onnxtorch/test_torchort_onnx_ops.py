"""
@brief      test log(time=3s)
"""
import unittest
import pprint
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
import numpy
from onnx.numpy_helper import to_array
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from skl2onnx import to_onnx
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxMatMul, OnnxAdd, OnnxSoftmax)
import torch
from deeponnxcustom import __max_supported_opset__ as TARGET_OPSET
from deeponnxcustom.tools.onnx_helper import onnx_rename_weights
from deeponnxcustom.onnxtorch import TorchOrtFactory


class TestTorchOrtOnnxOps(ExtTestCase):

    def get_onnx_graph(self, name):
        if name == "reg":
            data = load_diabetes()
            X, y = data.data, data.target  # pylint: disable=E1101
            y /= 100
            nn = MLPRegressor(hidden_layer_sizes=(20,), max_iter=1)
            nn.fit(X, y)
            nn_onnx = to_onnx(nn, X[1:].astype(numpy.float32),
                              target_opset=TARGET_OPSET)
            self.assertEqual(len(nn_onnx.graph.output), 1)
            onnx_rename_weights(nn_onnx)
            weights = [(init.name, to_array(init))
                       for init in nn_onnx.graph.initializer
                       if 'shape' not in init.name]
            return nn_onnx, weights, X, y, None

        if name == "softmax":

            class CustomSoftmax(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, weights, intercept):
                    y = torch.softmax(x @ weights + intercept, 1)
                    ctx.save_for_backward(x, weights, intercept, y)
                    return y

                @staticmethod
                def backward(ctx, grad_output):
                    x, weights, intercept, y = ctx.saved_tensors
                    return torch.autograd.grad(
                        y, [x, weights, intercept], grad_outputs=grad_output)

            X = numpy.random.randn(100, 4).astype(numpy.float32)
            self.assertEqual(X.shape, (100, 4))
            y = X.sum(axis=1) + numpy.random.randn(100) / 10
            y = y.astype(numpy.float32)
            self.assertEqual(y.shape, (100, ))
            weights = numpy.random.randn(4, 1).astype(numpy.float32)
            intercept = numpy.random.randn(1).astype(numpy.float32)

            node = OnnxSoftmax(
                OnnxAdd(
                    OnnxMatMul('X', weights, op_version=TARGET_OPSET),
                    intercept, op_version=TARGET_OPSET),
                op_version=TARGET_OPSET)
            nn_onnx = node.to_onnx({'X': X}, target_opset=TARGET_OPSET)
            self.assertEqual(len(nn_onnx.graph.output), 1)
            onnx_rename_weights(nn_onnx)
            weights = [(init.name, to_array(init))
                       for init in nn_onnx.graph.initializer]
            return nn_onnx, weights, X, y, CustomSoftmax

        raise AssertionError("Unexpected value %r." % name)

    def common_onnx_graph(self, name, device, n_iter=1, debug=False):
        test_data = self.get_onnx_graph(name)
        onnx_graph, weights, X_train, y_train, torch_fct = test_data

        # onnx part
        fact = TorchOrtFactory(
            onnx_graph, [w[0] for w in weights], providers=device)
        cls_ort = fact.create_class(keep_models=True, debug=debug)
        cls_tch = torch_fct

        # torch part

        if debug:
            pprint.pprint(cls_ort.__dict__)

        def from_numpy(v, device=None, requires_grad=False):
            v = torch.from_numpy(v)
            if device is not None:
                v = v.to(device)
            v.requires_grad_(requires_grad)
            return v

        def train_cls(cls, device, X_train, y_train, weights,
                      n_iter=n_iter, learning_rate=1e-2):
            x = from_numpy(X_train.astype(numpy.float32),
                           requires_grad=True, device=device)
            y = from_numpy(y_train.astype(numpy.float32),
                           requires_grad=True, device=device)

            weights_tch = [(w[0], from_numpy(w[1], requires_grad=True, device=device))
                           for w in weights]
            weights_values = [w[1] for w in weights_tch]

            all_losses = []
            all_grads = []
            for t in range(n_iter):
                # forward - backward
                y_pred = cls.apply(x, *weights_values)
                loss = (y_pred - y).pow(2).sum()
                loss.backward()

                # update weights
                with torch.no_grad():
                    grads = []
                    for _, w in weights_tch:
                        grads.append(w.grad.cpu().numpy())
                        w -= w.grad * learning_rate
                        w.grad.zero_()
                    all_grads.append(grads)

                all_losses.append((t, float(loss.detach().numpy())))
            return all_losses, weights_tch, all_grads

        ort_train_losses, ort_final_weights, ort_all_grads = train_cls(
            cls_ort, device, X_train, y_train, weights, n_iter=n_iter)
        if cls_tch is not None:
            tch_train_losses, tch_final_weights, tch_all_grads = train_cls(
                cls_tch, device, X, y, weights, n_iter=n_iter)
            self.assertEqual(tch_all_grads, ort_all_grads)
            self.assertEqual(tch_final_weights, ort_final_weights)
            self.assertEqual(tch_train_losses, ort_train_losses)

    @ignore_warnings(ConvergenceWarning)
    def test_onnx_ops(self):
        for name in ['softmax', 'reg']:
            for device_name in ['cpu', 'cuda:0']:
                if device_name == 'cuda:0' and not torch.cuda.is_available():
                    continue
                with self.subTest(name=name, device=device_name):
                    device = torch.device(device_name)
                    self.common_onnx_graph(name, device)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('skl2onnx')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestTorchOrtOnnxOps().test_onnx_ops()
    unittest.main()
