"""
@file
@brief Helpers about ONNX.
"""
import torch


def save_as_onnx(model, filename, size, target_opset=14,
                 batch_size=1, device='cpu',
                 keep_initializers_as_inputs=False):
    """
    Converts a torch model into ONNX using
    :func:`torch.onnx.export`.

    :param model: torch model
    :param filename: output filename
    :param size: input size
    :param target_opset: opset to use for the conversion
    :param batch_size: batch size
    :param device: device
    :param keep_initializers_as_inputs: see :func:`torch.onnx.export`
    """
    size = (batch_size, ) + (size, )
    x = torch.randn(  # pylint: disable=E1101
        size, requires_grad=True).to(device)
    torch.onnx.export(
        model, x, filename,
        do_constant_folding=False,
        export_params=False,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}})
