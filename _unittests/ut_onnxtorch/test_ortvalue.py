"""
@brief      test log(time=5s)
"""
import unittest
import copy
import gc
import time
import numpy
from numpy.testing import assert_almost_equal
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue, OrtValueVector)
from onnxruntime.training.ortmodule import ORTModule
import torch
from torch._C import _from_dlpack
from torch.utils.dlpack import from_dlpack
try:
    from cpyquickhelper.profiling._event_profiler_c import (  # pylint: disable=E0611
        get_memory_content)
except ImportError:
    get_memory_content = None


class TestOrtValue(ExtTestCase):

    # DLPack structure:
    # https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L144
    capsule_size = 48

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

    def test_torch_dlpack(self):
        numpy_arr_input = numpy.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
        tensor = torch.from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(tensor.shape))
        ptr = tensor.data_ptr()

        dlp = tensor.__dlpack__()
        if get_memory_content is not None:
            dlp_content = get_memory_content(
                dlp, 'dltensor', TestOrtValue.capsule_size)
        tensor2 = _from_dlpack(dlp)
        self.assertEqual(ptr, tensor2.data_ptr())
        new_array = tensor2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        device = tensor.__dlpack_device__()
        self.assertEqual((1, 0), device)

        dlp2 = tensor2.__dlpack__()
        if get_memory_content is not None:
            dlp2_content = get_memory_content(
                dlp2, 'dltensor', TestOrtValue.capsule_size)
            self.assertNotEqual(dlp_content, dlp2_content)

    def test_ortvalue_potential_bug(self):
        numpy_arr_input = numpy.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
        dlps = []
        for _ in range(10000):
            ov = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
            dlps.append(ov._ortvalue.to_dlpack())
            del ov

        gc.collect()
        time.sleep(0.5)
        tensors = []
        for dlp in dlps:
            tensors.append(from_dlpack(dlp))
        del dlps
        gc.collect()
        time.sleep(0.5)

        for t in tensors:
            new_array = t.numpy()
            assert_almost_equal(numpy_arr_input, new_array)

    def test_ortvalue_dlpack(self):
        numpy_arr_input = numpy.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        if get_memory_content is not None:
            dlp_content = get_memory_content(
                dlp, 'dltensor', TestOrtValue.capsule_size)
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        if get_memory_content is not None:
            dlp2_content = get_memory_content(
                dlp, 'dltensor', TestOrtValue.capsule_size)
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

        if get_memory_content is not None:
            dlp3 = ortvalue2.__dlpack__()
            dlp3_content = get_memory_content(
                dlp3, 'dltensor', TestOrtValue.capsule_size)
            self.assertEqual(dlp_content, dlp2_content)
            self.assertNotEqual(dlp_content, dlp3_content)
            self.assertNotEqual(dlp3_content, dlp2_content)

    def test_ortvalue_vector(self):
        narrays = [
            numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        dtype=numpy.float32),
            numpy.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=numpy.float32)]
        vect = OrtValueVector()
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a)
            vect.push_back(ortvalue._ortvalue)
        self.assertEqual(len(vect), 2)
        for ov, ar in zip(vect, narrays):
            ovar = ov.numpy()
            assert_almost_equal(ar, ovar)

    def test_ortvalue_vector_dlpack(self):
        narrays = [
            numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        dtype=numpy.float32),
            numpy.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=numpy.float32)]
        vect = OrtValueVector()
        ptr = []
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a)
            vect.push_back(ortvalue._ortvalue)
            ptr.append(ortvalue.data_ptr())
        self.assertEqual(len(vect), 2)

        def my_to_tensor(dlpack_structure):
            return C_OrtValue.from_dlpack(dlpack_structure, False)

        ortvalues = vect.to_dlpack(my_to_tensor)
        self.assertEqual(len(ortvalues), len(vect))

        ptr2 = []
        for av1, v2 in zip(narrays, ortvalues):
            ptr2.append(v2.data_ptr())
            av2 = v2.numpy()
            assert_almost_equal(av1, av2)
        self.assertEqual(ptr, ptr2)

    def ortmodule_dlpack(self, device):

        class NeuralNetTanh(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NeuralNetTanh, self).__init__()

                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.tanh = torch.nn.Tanh()

            def forward(self, input1):
                out = self.fc1(input1)
                out = self.tanh(out)
                return out

        def run_step(model, x):
            prediction = model(x)
            loss = prediction.sum()
            loss.backward()
            return prediction, loss

        N, D_in, H, D_out = 120, 1536, 500, 1536
        pt_model = NeuralNetTanh(D_in, H, D_out).to(device)
        ort_model = ORTModule(copy.deepcopy(pt_model))

        for _ in range(10):
            pt_x = torch.randn(N, D_in, device=device, requires_grad=True)
            ort_x = copy.deepcopy(pt_x)
            ort_prediction, ort_loss = run_step(ort_model, ort_x)
            pt_prediction, pt_loss = run_step(pt_model, pt_x)
            self.assert_values_are_close(
                ort_prediction, pt_prediction, atol=1e-4)
            self.assert_values_are_close(ort_x.grad, pt_x.grad)
            self.assert_values_are_close(ort_loss, pt_loss, atol=1e-4)

    @ignore_warnings(UserWarning)
    def test_ortmodule_dlpack(self):
        for device_name in ['cuda:0', 'cpu']:
            if device_name == 'cuda:0' and not torch.cuda.is_available():
                continue
            with self.subTest(device=device_name):
                device = torch.device(device_name)
                self.ortmodule_dlpack(device)


if __name__ == "__main__":
    unittest.main()
