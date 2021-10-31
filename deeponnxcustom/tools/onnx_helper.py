"""
@file
@brief Helpers about ONNX.
"""
import math


def save_as_onnx(model, filename, size=None, target_opset=14,
                 batch_size=1, device='cpu',
                 keep_initializers_as_inputs=False):
    """
    Converts a torch model into ONNX using
    :func:`torch.onnx.export`. The function works
    on models with only one input.

    :param model: torch model
    :param filename: output filename
    :param size: input size or left None to guess it from the model
    :param target_opset: opset to use for the conversion
    :param batch_size: batch size
    :param device: device
    :param keep_initializers_as_inputs: see :func:`torch.onnx.export`
    """
    import torch  # pylint: disable=C0415

    if size is None:
        for p in model.named_parameters():
            name, value = p
            if name.endswith('weight'):
                size = value.shape[-1]
                break
        if size is None:
            raise RuntimeError(
                "Unable to guess size from the following list of "
                "parameters:\n%s" % ("\n".join(
                    "%r: shape=%r - dtype=%r" % (name, tuple(v.shape), v.dtype)
                    for name, v in model.named_parameters())))

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


def onnx_rename_weights(onx):
    """
    Renames ONNX initialiers to make sure their name
    follows the alphabetical order. The model is
    modified inplace. This function calls
    :func:`onnx_rename_names
    <mlprodict.onnx_tools.onnx_manipulations.onnx_rename_names>`.

    :param onx: ONNX model
    :return: same model

    .. note::
        The function does not go into subgraphs.
    """
    from mlprodict.onnx_tools.onnx_manipulations import (  # pylint: disable=C0415
        onnx_rename_names)

    init = [init.name for init in onx.graph.initializer]
    ninit = max(1, int(math.log(len(init)) / math.log(10) + 1))
    fmt = "I%0{}d_%s".format(ninit)
    new_names = [fmt % (i, name) for i, name in enumerate(init)]
    repl = dict(zip(init, new_names))
    return onnx_rename_names(onx, recursive=False, replace=repl)
