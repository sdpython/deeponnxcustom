"""
@brief      test log(time=3s)
"""
import unittest
import pprint
from pyquickhelper.pycode import ExtTestCase
import numpy
from onnx.numpy_helper import to_array
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from skl2onnx import to_onnx
import torch
from deeponnxcustom.tools.onnx_helper import onnx_rename_weights
from deeponnxcustom.onnxtorch import TorchOrtFactory


class TestTorchOrtExtended(ExtTestCase):

    def common_mlp_regressor(self, device, debug=False):
        data = load_diabetes()
        X, y = data.data, data.target  # pylint: disable=E1101
        y /= 100
        X_train, _, y_train, y_test = train_test_split(X, y)
        nn = MLPRegressor(hidden_layer_sizes=(20,), max_iter=2000)
        nn.fit(X_train, y_train)
        nn_onnx = to_onnx(nn, X_train[1:].astype(numpy.float32))

        onnx_rename_weights(nn_onnx)
        weights = [(init.name, to_array(init))
                   for init in nn_onnx.graph.initializer
                   if 'shape' not in init.name]
        fact = TorchOrtFactory(nn_onnx, [w[0] for w in weights])
        cls = fact.create_class()
        if debug:
            pprint.pprint(cls.__dict__)

        def from_numpy(v, device=None, requires_grad=False):
            v = torch.from_numpy(v)
            if device is not None:
                v = v.to(device)
            v.requires_grad_(requires_grad)
            return v

        def train_cls(cls, device, X_train, y_train, weights, n_iter=20, learning_rate=1e-2):
            x = from_numpy(X_train.astype(numpy.float32),
                           requires_grad=True, device=device)
            y = from_numpy(y_train.astype(numpy.float32),
                           requires_grad=True, device=device)

            weights_tch = [(w[0], from_numpy(w[1], requires_grad=True, device=device))
                           for w in weights]
            weights_values = [w[1] for w in weights_tch]

            all_losses = []
            for t in range(n_iter):
                # forward - backward
                y_pred = cls.apply(x, *weights_values)
                loss = (y_pred - y).pow(2).sum()
                loss.backward()

                # update weights
                with torch.no_grad():
                    for _, w in weights_tch:
                        w -= w.grad * learning_rate
                        w.grad.zero_()

                all_losses.append((t, float(loss.detach().numpy())))
            return all_losses, weights_tch

        train_losses, final_weights = train_cls(
            cls, device, X_train, y_test, weights)
        self.assertNotEmpty(train_losses)
        self.assertNotEmpty(final_weights)

    def test_mlp_regressor_gpu(self):
        if not torch.cuda.is_available():
            return
        device_name = "cuda:0"
        device = torch.device(device_name)
        self.common_mlp_regressor(device)

    def test_mlp_regressor_cpu(self):
        device_name = "cpu"
        device = torch.device(device_name)
        self.common_mlp_regressor(device, debug=False)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('deeponnxcustom')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestTorchOrtExtended().test_mlp_regressor_cpu()
    unittest.main()
