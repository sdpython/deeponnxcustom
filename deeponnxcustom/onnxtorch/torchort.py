"""
@file
@brief Experimental.
"""
import warnings
import logging
from textwrap import dedent
from io import BytesIO
import onnx
from onnxruntime import InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding)
try:
    from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
        TrainingAgent, OrtValueCache, OrtModuleGraphBuilder,
        OrtModuleGraphBuilderConfiguration, OrtDevice,
        TrainingGraphTransformerConfiguration, OrtValueVector,
        PartialGraphExecutionState)
except ImportError:  # pragma: no cover
    # onnxruntime-training is not installed.
    warnings.warn(
        "TorchOrtFactory cannot work without onnxruntime-training.")
from onnxruntime import RunOptions
from torch import is_grad_enabled  # pylint: disable=E0611
from torch.autograd import Function
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch._C import _from_dlpack


class TorchOrtFunction(Function):
    """
    Ancestor to all classes created by @see cl TorchOrtFactory.
    It implements simple functions to move the ownership of tensors
    from *onnxruntime* to *pytorch* (or the other way around)
    through :epkg:`DLPack` structures.
    """

    @staticmethod
    def from_torch_to_ort(tensors):
        "Converts a list of pytorch tensors into an OrtValueVector."
        vect = OrtValueVector()
        vect.reserve(len(tensors))
        for t in tensors:
            if t is None:
                # if gradient then
                # grad_output = torch.zeros(shape, device=device, dtype=dtype)
                raise NotImplementedError(  # pragma: no cover
                    "Empty vector found.")
            if not t.is_contiguous():
                # grad = grad.contiguous()
                raise NotImplementedError(  # pragma: no cover
                    "Non contiguous gradient found.")
            vect.push_back(to_dlpack(t), False)
        return vect

    @staticmethod
    def from_ort_to_torch(ort_values):
        "Converts a OrtValueVector into a tuple of pytorch tensors."
        # return tuple(_from_dlpack(ov.to_dlpack()) for ov in ort_values)
        if hasattr(ort_values, 'to_dlpack'):
            return tuple(ort_values.to_dlpack(_from_dlpack))
        if len(ort_values) == 0:
            raise RuntimeError(  # pragma: no cover
                "The conversion fails on an empty vector.")
        if hasattr(ort_values[0], '__dlpack__'):
            return tuple(  # pragma: no cover
                from_dlpack(ov) for ov in ort_values)
        else:
            return tuple(_from_dlpack(ov.to_dlpack()) for ov in ort_values)


def ort_forward(ctx, *inputs):
    """
    Implements forward function.
    See :epkg:`autograd functions`.
    """
    cls = ctx._forward_cls
    logger = cls._logger
    training = is_grad_enabled() or any(ctx.needs_input_grad)

    def _log(msg):
        logger.debug("[%s.forward] (%dI) %s" % (
            cls.__name__, len(inputs), msg))

    if logger is not None:
        if training:
            _log("begin with gradient")
        else:
            _log("begin")
        _log("torch function %r" % type(ctx))
        _log("ort class %r" % cls)
        _log("create OrtValueVector (through dlpack)")

    forward_inputs = cls.from_torch_to_ort(inputs)

    if training:
        forward_outputs = OrtValueVector()
        state = PartialGraphExecutionState()
        cls._states.append(state)
        if logger is not None:
            _log("run_forward")
        cls._training_agent.run_forward(
            forward_inputs, forward_outputs, state, cls._cache)

        ctx.save_for_backward(*inputs)

        if cls._update_cache:
            if logger is not None:
                _log("update_cache")
            raise NotImplementedError("Cache is not implemented.")

            # for i in range(self._cache_start, len(forward_outputs)):
            #     self.cache.insert(
            #         self._cached_node_arg_names[i - cls._cache_start],
            #         forward_outputs[i])
            # self._update_cache = False
            # if logger is not None:
            #     _log("to torck.tensor")
            # return tuple(_utils._ortvalue_to_torch_tensor
            # (forward_outputs[i], device) for i in range(self._cache_start))

        else:
            if logger is not None:
                _log("to torck.tensor")
            res = cls.from_ort_to_torch(forward_outputs)
            if len(res) == 1:
                res = res[0]
            if logger is not None:
                _log("end")
            return res
    else:
        # what about bind_input (+ data_ptr)
        if len(forward_inputs) != len(cls._grad_input_names):
            raise RuntimeError(  # pragma: no cover
                "Size mismatch len(inputs)=%d, len(onnx inputs)=%d." % (
                    len(forward_inputs), len(cls._grad_input_names)))
        iobinding = SessionIOBinding(cls._sess_eval._sess)
        if logger is not None:
            _log("bind inputs %r" % cls._grad_input_names)
        for name, inp in zip(
                cls._grad_input_names, forward_inputs):
            iobinding.bind_ortvalue_input(name, inp)

        # bind output
        if logger is not None:
            _log("bind outputs %r" % cls._output_names)
        for name, dev in zip(
                cls._output_names, cls._fw_no_grad_output_device_info):
            iobinding.bind_output(name, dev)

        # if the shape is known in advance
        # iobinding.bind_output(
        #    output_desc.name, torch_tensor.device.type,
        #    _utils.get_device_index(target_device),
        #    _utils.dtype_torch_to_numpy(torch_tensor.dtype),
        #    list(torch_tensor.size()), torch_tensor.data_ptr())

        if logger is not None:
            _log("grad_enabled=False (run_with_iobinding)")
        cls._sess_eval._sess.run_with_iobinding(iobinding, cls._run_options)
        if logger is not None:
            _log("get_outputs")
        ortvalues = iobinding.get_outputs()
        if logger is not None:
            _log("to torck.tensor")
        res = cls.from_ort_to_torch(ortvalues)
        if len(res) == 1:
            res = res[0]
        if logger is not None:
            _log("end")
        return res


def ort_backward(ctx, *grad_outputs):
    """
    Implements backward function.
    See :epkg:`autograd functions`.
    """
    cls = ctx._forward_cls
    logger = cls._logger

    def _log(msg):
        logger.debug("[%s.backward] (%dI) %s" % (
            cls.__name__, len(grad_outputs), msg))

    if logger is not None:
        _log("begin")
        _log("torch function %r" % type(ctx))
        _log("ort class %r" % cls)
        _log("saved_tensors")

    inputs = ctx.saved_tensors
    if cls._debug:
        print(  # pragma: no cover
            "DEBUG: saved_tensors %r" % type(inputs))
    if logger is not None:
        _log("cls._state.pop()")
    state = cls._states.pop()

    if logger is not None:
        _log("create OrtValueVector (through dlpack)")

    backward_inputs = cls.from_torch_to_ort(grad_outputs)

    backward_outputs = OrtValueVector()
    if logger is not None:
        _log("run_backward")
    cls._training_agent.run_backward(backward_inputs, backward_outputs, state)
    res = cls.from_ort_to_torch(backward_outputs)
    if len(res) == 1:
        res = res[0]
    else:
        if cls._debug:  # pragma: no cover
            print("DEBUG")
            for i, ov in enumerate(backward_outputs):
                print("BCK-RET: i=%d - ptr=%r - shape=%r" % (
                    i, ov.shape(), ov.data_ptr()))
        if logger is not None:
            _log("got %r gradients" % len(res))
    if logger is not None:
        _log("end")
    return res


class TorchOrtFactory:
    """
    A class which dynamically another class which implements a
    custom function (see :epkg:`autograd functions`).
    Use ONNX inside a torch function. Only initializers
    can be trained, no parameters.

    :param onnx_model: onnx model
    :param weights_to_train: names of the weights to train
    :param input_names: input names or None for all
    :param output_names: output names or None for all
    :param class_name: class name
    :param sess_options: see :epkg:`SessionOptions`
    :param providers: see :epkg:`InferenceSession`
    :param provider_options: see :epkg:`InferenceSession`
    :param run_options: see :epkg:`RunOptions`
    :param graph_builder_config:
        see :epkg:`OrtModuleGraphBuilderConfiguration`
    :param device_index: used for cuda (0 for `cuda:0`,
        `cuda:1`, ...), 0 by default

    .. note::
        The current implementation of :epkg:`onnxruntime` forces
        the weights to train to appear in the alphabetical order.
        The constructor checks that condition is verified.

    .. warning::
        This class does not consider subgraphs.
    """

    def __init__(self, onnx_model, weights_to_train,
                 input_names=None, output_names=None,
                 class_name=None,
                 sess_options=None, providers=None,
                 provider_options=None, run_options=None,
                 graph_builder_config=None,
                 device_index=0):
        self.onnx_model = onnx_model
        self.input_names = input_names
        self.output_names = output_names
        self.class_name = class_name
        self.weights_to_train = weights_to_train
        self.device_index = device_index

        self.provider_options = provider_options
        self.sess_options = sess_options
        self.providers = providers
        self.run_options = run_options
        self.graph_builder_config = graph_builder_config

        # default
        if self.weights_to_train is None:
            raise ValueError(  # pragma: no cover
                "weights_to_train must be specified.")
        if self.input_names is None:
            self.input_names = [obj.name
                                for obj in self.onnx_model.graph.input]
        if self.output_names is None:
            self.output_names = [obj.name
                                 for obj in self.onnx_model.graph.output]
        if self.class_name is None:
            self.class_name = "TorchOrtFunction_%r" % id(self)
        if self.providers in (None, 'cpu'):
            self.providers = ["CPUExecutionProvider" for i in self.input_names]
            if self.provider_options is None:
                self.provider_options = [{} for i in self.input_names]
        if self.run_options is None:
            self.run_options = RunOptions()
            self.run_options.training_mode = True

        if len(self.input_names) != len(self.providers):
            raise ValueError(  # pragma: no cover
                "input_names and providers must have the same length.")
        if len(self.input_names) != len(self.provider_options):
            raise ValueError(  # pragma: no cover
                "input_names and provider_options must have the same length.")

        if list(sorted(self.weights_to_train)) != self.weights_to_train:
            raise ValueError(
                "List of weights to train must be sorted but is not in %r. "
                "You shoud use function onnx_rename_weights to do that "
                "before calling this class." % self.weights_to_train)

        if self.graph_builder_config is None:
            initializer_names = [
                i.name for i in self.onnx_model.graph.initializer]
            input_names = [i.name for i in self.onnx_model.graph.input]

            config = OrtModuleGraphBuilderConfiguration()
            config.initializer_names = initializer_names
            config.initializer_names_to_train = self.weights_to_train
            config.input_names_require_grad = input_names
            config.build_gradient_graph = True

            p = TrainingGraphTransformerConfiguration()
            config.graph_transformer_config = p

            # config.enable_caching = True
            # config.loglevel =
            # config.use_memory_efficient_gradient = True
            self.graph_builder_config = config

    def __repr__(self):
        "usual"
        return "%s(...)" % self.__class__.__name__

    @staticmethod
    def _repr_helper_(obj, indent=0):
        "used to improve logging messages"
        if obj is None:
            return 'None'
        rows = []
        for c in sorted(dir(obj)):
            if c[0] == '_':
                continue
            try:
                value = getattr(obj, c)
            except AttributeError:  # pragma: no cover
                continue
            rows.append("%s=%r" % (c, value))

        if indent == 0:
            return "%s(%s)" % (obj.__class__.__name__, ", ".join(rows))
        return "%s(\n    %s)" % (
            obj.__class__.__name__,
            "\n    ".join(rows))

    @staticmethod
    def _provider_name_to_device_type(provider_name):
        if provider_name == 'CPUExecutionProvider':
            return OrtDevice.cpu()
        if provider_name == 'GPUExecutionProvider':  # pragma: no cover
            return OrtDevice.cuda()
        raise ValueError(  # pragma: no cover
            'Unexpected provider name %r.' % provider_name)

    def create_class(self, enable_logging=False, keep_models=False,
                     debug=False):
        """
        Creates a class which inherits from
        :func:`torch.autograd.Function` and implements forward,
        backward methods using ONNX. The function dynamically
        creates a new class and pushes every needed objects
        as static attributes of the new class.

        :param enable_logging: used to debug, logs every building step,
            at info level, logs information while processing forward
            and backward at debug level
        :param keep_models: stores additional information as
            static attributes
        :param debug: display information
        :return: a new class

        The pattern follows the documentation described in
        :epkg:`autograd functions`. Methods forward and backward
        are replaced by onnx implementations, runtime is
        :epkg:`onnxruntime-training`.

        ::

            class CustomClass(torch.autograd.Function):

                @staticmethod
                def forward(ctx, *input):
                    ctx.save_for_backward(*input)
                    return ...

                @staticmethod
                def backward(ctx, *grad_output):
                    input, = ctx.saved_tensors
                    grad_input = grad_output.clone()
                    grad_input[input < 0] = 0
                    return grad_input
        """
        if enable_logging:
            logger = logging.getLogger("deeponnxcustom")
        else:
            logger = None

        doc = dedent("""Use onnxruntime to compute the gradient
                        in a pytorch function.""")

        if logger is not None:
            logger.info("[TorchOrtFactory] create training onnx")
            logger.info("[TorchOrtFactory] input_names=%r",
                        self.input_names)
            logger.info("[TorchOrtFactory] output_names=%r",
                        self.output_names)
            logger.info("[TorchOrtFactory] weights_to_train=%r",
                        self.weights_to_train)

        builder = OrtModuleGraphBuilder()

        if logger is not None:
            cf = self.graph_builder_config.graph_transformer_config
            cfp = cf.propagate_cast_ops_config
            logger.info("[TorchOrtFactory] OrtModuleGraphBuilder.initialize")
            logger.info(
                "[TorchOrtFactory] graph_builder_config=%s",
                TorchOrtFactory._repr_helper_(
                    self.graph_builder_config, indent=4))
            logger.info(
                "[TorchOrtFactory] graph_builder_config."
                "graph_transformer_config=%s",
                TorchOrtFactory._repr_helper_(cf, indent=4))
            logger.info(
                "[TorchOrtFactory] graph_builder_config."
                "graph_transformer_config.propagate_cast_ops_config=%s",
                TorchOrtFactory._repr_helper_(cfp, indent=4))

        builder.initialize(
            self.onnx_model.SerializeToString(),
            self.graph_builder_config)

        if logger is not None:
            logger.info("[TorchOrtFactory] OrtModuleGraphBuilder.build")
        builder.build()

        if logger is not None:
            logger.info("[TorchOrtFactory] OrtModuleGraphBuilder.get_model")

        train_onnx_model_serialized = builder.get_model()

        optimized_pre_grad_model = builder.get_inference_optimized_model()
        graph_info = builder.get_graph_info()

        if logger is not None:
            logger.info("[TorchOrtFactory] graph_info=%s",
                        TorchOrtFactory._repr_helper_(
                            graph_info, indent=4))
            logger.info("[TorchOrtFactory] create TrainSession")
            logger.info("[TorchOrtFactory] sess_options=%s",
                        TorchOrtFactory._repr_helper_(
                            self.sess_options, indent=4))
            logger.info("[TorchOrtFactory] providers=%r", self.providers)

        sess = InferenceSession(
            train_onnx_model_serialized,
            sess_options=self.sess_options,
            provider_options=self.provider_options,
            providers=self.providers)

        if logger is not None:
            logger.info("[TorchOrtFactory] create InferenceSession")

        sess_eval = InferenceSession(
            optimized_pre_grad_model,
            sess_options=self.sess_options,
            provider_options=self.provider_options,
            providers=self.providers)

        if logger is not None:
            logger.info("[TorchOrtFactory] create training agent")

        grad_input_names = [obj.name for obj in sess.get_inputs()]
        bw_fetches_names = [obj.name for obj in sess.get_outputs()]

        fw_outputs_device_info = [
            OrtDevice(
                TorchOrtFactory._provider_name_to_device_type(i),
                OrtDevice.default_memory(),
                self.device_index)
            for i in self.providers]
        bw_outputs_device_info = [
            OrtDevice(
                TorchOrtFactory._provider_name_to_device_type(
                    self.providers[0]),
                OrtDevice.default_memory(),
                self.device_index)
            for i in bw_fetches_names]
        fw_no_grad_output_device_info = [
            OrtDevice(
                TorchOrtFactory._provider_name_to_device_type(
                    self.providers[0]),
                OrtDevice.default_memory(),
                self.device_index)
            for i in self.output_names]

        training_agent = TrainingAgent(
            sess._sess,
            grad_input_names,
            fw_outputs_device_info,
            bw_fetches_names,
            bw_outputs_device_info)

        if logger is not None:
            logger.info(
                "[TorchOrtFactory] instantiate dynamic class %r",
                self.class_name)
            logger.info(
                "[TorchOrtFactory] weights_to_train=%r",
                self.weights_to_train)
            logger.info(
                "[TorchOrtFactory] grad_input_names=%r",
                grad_input_names)
            logger.info(
                "[TorchOrtFactory] bw_fetches_names=%r",
                bw_fetches_names)
            logger.info(
                "[TorchOrtFactory] device_index=%r",
                self.device_index)

        kwargs = {
            '__doc__': doc,
            '__module__': __name__,
            '_run_options': self.run_options,
            '_sess': sess,
            '_sess_eval': sess_eval,
            '_training_agent': training_agent,
            '_cache': OrtValueCache(),
            '_update_cache': False,
            '_states': [],
            '_logger': logger,
            '_input_names': self.input_names,
            '_debug': debug,
            '_grad_input_names': grad_input_names,
            '_output_names': self.output_names,
            '_bw_fetches_names': bw_fetches_names,
            '_fw_outputs_device_info': fw_outputs_device_info,
            '_bw_outputs_device_info': bw_outputs_device_info,
            '_fw_no_grad_output_device_info': fw_no_grad_output_device_info,
            '_weights_to_train': list(sorted(
                self.weights_to_train)),
            'forward': staticmethod(ort_forward),
            'backward': staticmethod(ort_backward),
            '_graph_info': graph_info}

        if keep_models:
            kwargs.update(dict(
                _trained_onnx=onnx.load(BytesIO(train_onnx_model_serialized)),
                _optimized_pre_grad_model=onnx.load(
                    BytesIO(optimized_pre_grad_model)),
                _graph_builder=builder,
                _factory=self))

        newclass = type(self.class_name, (TorchOrtFunction,), kwargs)
        return newclass
