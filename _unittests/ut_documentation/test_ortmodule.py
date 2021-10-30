"""
@brief      test log(time=600s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase


class TestOrtModule(ExtTestCase):

    def test_import_ortmodule(self):

        from onnxruntime.training import ORTModule
        self.assertNotEmpty(ORTModule)

    def test_ortmodule(self):

        from onnxruntime.training import ORTModule
        import torch

        class NeuralNetwork(torch.nn.Module):
            def __init__(self):
                super(NeuralNetwork, self).__init__()
                self.flatten = torch.nn.Flatten()
                self.linear_relu_stack = torch.nn.Sequential(
                    torch.nn.Linear(28 * 28, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 10))

            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits

        mod = ORTModule(NeuralNetwork())
        self.assertNotEmpty(mod)


if __name__ == "__main__":
    unittest.main()
