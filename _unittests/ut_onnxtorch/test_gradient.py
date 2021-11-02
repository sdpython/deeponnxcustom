"""
@brief      test log(time=3s)
"""
import unittest
import copy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
import torch
from onnxruntime.training.ortmodule import ORTModule


class TestGradient(ExtTestCase):

    def assert_values_are_close(self, tensor, other, rtol=1e-05, atol=1e-06):
        are_close = torch.allclose(tensor, other, rtol=rtol, atol=atol)
        if not are_close:
            abs_diff = torch.abs(tensor - other)
            abs_other = torch.abs(other)
            max_atol = torch.max((abs_diff - rtol * abs_other))
            max_rtol = torch.max((abs_diff - atol) / abs_other)
            raise AssertionError(
                "The maximum atol is %r, maximum rtol is %r." % (
                    max_atol, max_rtol))

    def assert_gradients_match_and_reset_gradient(
            self, ort_model, pt_model, none_pt_params=None,
            reset_gradient=True, rtol=1e-05, atol=1e-06):
        if none_pt_params is None:
            none_pt_params = []
        ort_named_params = list(ort_model.named_parameters())
        pt_named_params = list(pt_model.named_parameters())
        self.assertEqual(len(ort_named_params), len(pt_named_params))

        for ort_named_param, pt_named_param in zip(ort_named_params, pt_named_params):
            ort_name, ort_param = ort_named_param
            pt_name, pt_param = pt_named_param

            self.assertIn(pt_name, ort_name)
            if pt_name in none_pt_params:
                self.assertNotEmpty(pt_param.grad)
                if ort_param is not None:
                    self.assertFalse(torch.is_nonzero(
                        torch.count_nonzero(ort_param.grad)))
            else:
                self.assert_values_are_close(
                    ort_param.grad, pt_param.grad, rtol=rtol, atol=atol)

            if reset_gradient:
                ort_param.grad = None
                pt_param.grad = None

    class NeuralNetSinglePositionalArgument(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            torch.nn.Module.__init__(self)
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    def gradient_correctness(self, device):
        N, D_in, H, D_out = 32, 128, 500, 10
        pt_model = TestGradient.NeuralNetSinglePositionalArgument(
            D_in, H, D_out).to(device)
        ort_model = ORTModule(copy.deepcopy(pt_model))

        def run_step(model, x):
            prediction = model(x)
            loss = prediction.sum()
            loss.backward()
            return prediction

        for _ in range(10):
            x = torch.randn(N, D_in, device=device)
            pt_prediction = run_step(pt_model, x)
            ort_prediction = run_step(ort_model, x)

            self.assert_values_are_close(ort_prediction, pt_prediction)
            self.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

    @ignore_warnings(UserWarning)
    def test_gradient_correctness(self):
        for device_name in ['cuda:0', 'cpu']:
            if device_name == 'cuda:0' and not torch.cuda.is_available():
                continue
            with self.subTest(device=device_name):
                device = torch.device(device_name)
                self.gradient_correctness(device)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('deeponnxcustom')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestTorchOrt().test_gradient_order()
    unittest.main()
