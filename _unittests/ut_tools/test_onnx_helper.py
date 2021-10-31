"""
@brief      test log(time=3s)
"""
import unittest
import os
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxRelu, OnnxMatMul)
import torch
import torch.nn.functional as F
from deeponnxcustom.tools.onnx_helper import save_as_onnx, onnx_rename_weights


class TestOnnxHelper(ExtTestCase):

    def test_save_as_onnx(self):

        class CustomNet(torch.nn.Module):
            def __init__(self, n_features, hidden_layer_sizes, n_output):
                super(CustomNet, self).__init__()
                self.hidden = []

                size = n_features
                for i, hid in enumerate(hidden_layer_sizes):
                    self.hidden.append(torch.nn.Linear(size, hid))
                    size = hid
                    setattr(self, "hid%d" % i, self.hidden[-1])
                self.hidden.append(torch.nn.Linear(size, n_output))
                setattr(self, "predict", self.hidden[-1])

            def forward(self, x):
                for hid in self.hidden:
                    x = hid(x)
                    x = F.relu(x)
                return x

        temp = get_temp_folder(__file__, "temp_save_as_onnx")
        filename = os.path.join(temp, "custom_net.onnx")
        model = CustomNet(5, (4, 3), 1)
        save_as_onnx(model, filename, 5)
        self.assertExists(filename)

    def test_save_as_onnx_none(self):

        class CustomNet(torch.nn.Module):
            def __init__(self, n_features, hidden_layer_sizes, n_output):
                super(CustomNet, self).__init__()
                self.hidden = []

                size = n_features
                for i, hid in enumerate(hidden_layer_sizes):
                    self.hidden.append(torch.nn.Linear(size, hid))
                    size = hid
                    setattr(self, "hid%d" % i, self.hidden[-1])
                self.hidden.append(torch.nn.Linear(size, n_output))
                setattr(self, "predict", self.hidden[-1])

            def forward(self, x):
                for hid in self.hidden:
                    x = hid(x)
                    x = F.relu(x)
                return x

        temp = get_temp_folder(__file__, "temp_save_as_onnx_none")
        filename = os.path.join(temp, "custom_net.onnx")
        model = CustomNet(5, (4, 3), 1)
        save_as_onnx(model, filename)
        self.assertExists(filename)

    def test_save_as_onnx_none_exc(self):

        class CustomNet(torch.nn.Module):
            def __init__(self, n_features, hidden_layer_sizes, n_output):
                super(CustomNet, self).__init__()
                self.hidden = []

                size = n_features
                for i, hid in enumerate(hidden_layer_sizes):
                    self.hidden.append(torch.nn.Linear(size, hid))
                    size = hid
                    setattr(self, "hid%d" % i, self.hidden[-1])
                self.hidden.append(torch.nn.Linear(size, n_output))
                setattr(self, "predict", self.hidden[-1])

            def forward(self, x):
                for hid in self.hidden:
                    x = hid(x)
                    x = F.relu(x)
                return x

            def named_parameters(self, prefix='', recurse=True):
                for name, value in super(CustomNet, self).named_parameters():
                    if name.endswith('.weight'):
                        continue
                    yield name, value

        model = CustomNet(5, (4, 3), 1)
        self.assertRaise(lambda: save_as_onnx(model, "temp_none.onnx"),
                         RuntimeError)

    def test_onnx_rename_weights(self):
        N, D_in, D_out, H = 3, 3, 3, 3
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

        onx = onnx_rename_weights(onx)
        names = [init.name for init in onx.graph.initializer]
        self.assertEqual(['I0_Ma_MatMulcst', 'I1_Ma_MatMulcst1'], names)


if __name__ == "__main__":
    unittest.main()
