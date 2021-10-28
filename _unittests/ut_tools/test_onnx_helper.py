"""
@brief      test log(time=3s)
"""
import unittest
import os
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
import torch
import torch.nn.functional as F
from deeponnxcustom.tools.onnx_helper import save_as_onnx


class TestOnnxHelper(ExtTestCase):

    def test_save_as_onnx(self):

        class CustomNet(torch.nn.Module):
            def __init__(self, n_features, hidden_layer_sizes, n_output):
                super(CustomNet, self).__init__()  # pylint: disable=R1725
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


if __name__ == "__main__":
    unittest.main()
