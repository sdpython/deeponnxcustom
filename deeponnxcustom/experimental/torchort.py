"""
@file
@brief Experimental.
"""
import logging
from textwrap import dedent
from onnxruntime import TrainingSession
from onnxruntime.capi._pybind_state import (
    TrainingAgent, OrtValueCache, OrtModuleGraphBuilder,
    OrtModuleGraphBuilderConfiguration,
    TrainingGraphTransformerConfiguration,
    PartialGraphExecutionState, TrainingParameters, RunOptions)
from torch import from_numpy, is_grad_enabled
from torch.autograd import Function
from torch.utils.dlpack import from_dlpack, to_dlpack


class TorchOrtFunction(Function):
    """
    Ancestor to all classes created by @see cl TorchOrtFactory.
    """
    
    @staticmethod
    def form_ort_to_torch(ort_value):
        packed = ort_value.to_dlpack()
        return from_dlpack(packed)


def ort_forward(cls, ctx, *inputs):
    """
    Implements forward function.
    See :epkg:`autograd functions`.
    """
    def _log(msg):
        logger.debug("[%s.forward] (%d inputs) %s" % (
            cls.__name__, len(inputs), msg))
    
    logger = cls._logger 
    if logger is not None:
        _log("conversion to dlpack")
    packed = [inp.to_dlpack() for inp in inputs]
    
    if logger is not None:
        _log("create OrtValueVector")
    forward_inputs = C.OrtValueVector()
    forward_inputs.reserve(len(inputs))
    for i in inputs:
        forward_inputs.push_back(i.to_dlpack())
        
    if is_grad_enabled():
        if logger is not None:
            _log("grad_enabled=True")
        forward_outputs = OrtValueVector()
        state = PartialGraphExecutionState()
        cls._state.append(state)
        if logger is not None:
            _log("run_forward")
        cls._training_agent.run_forward(forward_inputs, forward_outputs, state, cls._cache)
        
        if cls._update_cache:
            if logger is not None:
                _log("update_cache")
            raise NotImplelmentedError("Cache is not implemented.")

            for i in range(self._cache_start, len(forward_outputs)):
                self.cache.insert(
                    self._cached_node_arg_names[i-self._cache_start], forward_outputs[i])
            self._update_cache = False
            if logger is not None:
                _log("to torck.tensor")
            return tuple(_utils._ortvalue_to_torch_tensor(forward_outputs[i], device) for i in range(self._cache_start))
    
        else:
            if logger is not None:
                _log("to torck.tensor")
            res = tuple(cls.from_ort_to_torch(ov) for ov in ortvalues)
            if logger is not None:
                _log("end")
            return res
    else:
        if logger is not None:
            _log("grad_enabled=False (run_with_iobinding)")
        cls._sess.run_with_iobinding(iobinding, cls._run_options)
        if logger is not None:
            _log("get_outputs")
        ortvalues = iobinding.get_outputs()
        if logger is not None:
            _log("to torck.tensor")
        res = tuple(cls.from_ort_to_torch(ov) for ov in ortvalues)
        if logger is not None:
            _log("end")
        return res
 

def ort_backward(cls, ctx, *grad_outputs):
    """
    Implements backward function.
    See :epkg:`autograd functions`.
    """
    def _log(msg):
        logger.debug("[%s.backward] (%d inputs) %s" % (
            cls.__name__, len(inputs), msg))
    
    logger = cls._logger 
    if logger is not None:
        _log("saved_tensors")
    inputs, = ctx.saved_tensors
    if logger is not None:
        _log("cls._state.pop()")
    state = cls._state.pop()

    if logger is not None:
        _log("create OrtValueVector")
    backward_inputs = C.OrtValueVector()
    backward_inputs.reserve(len(inputs))
    for i in grad_outputs:
        if grad_output is None:
            raise NotImplementedError()
            grad_output = torch.zeros(shape, device=device, dtype=dtype)
        backward_inputs.push_back(i.to_dlpack())

    backward_outputs = OrtValueVector()
    if logger is not None:
        _log("run_backward")
    cls._training_agent.run_backward(backward_inputs, backward_outputs, state)
    res = tuple(cls.from_ort_to_torch(ov) for ov in backward_outputs)
    if logger is not None:
        _log("end")
    return res


class TorchOrtFactory:
    """
    A class which dynamically another class which implements a
    custom function (see :epkg:`autograd functions`).
    Use ONNX inside a torch function.

    :param onnx_model: onnx model
    :param weights_to_train: names of the weights to train
    :param input_names: input names or None for all
    :param output_names: output names or None for all
    :param class_name: class name
    :param sess_options: see :epkg:`SessionOptions`
    :param providers: see :epkg:`TrainingSession`
    :param provider_options: see :epkg:`TrainingSession`
    :param run_options: see :epkg:`RunOptions`
    :param graph_builder_config: see :epkg:`OrtModuleGraphBuilderConfiguration`
    """
    
    def __init__(self, onnx_model, weights_to_train,
                 input_names=None, output_names=None,
                 class_name=None, training_parameters=None,
                 sess_options=None, providers=None,
                 provider_options=None, run_options=None,
                 graph_builder_config=None):
        self.onnx_model = onnx_model
        self.input_names = input_names
        self.output_names = output_names
        self.class_name = class_name

        p = TrainingParameters()
        p.loss_output_name = "loss"
        p.weights_to_train = set(weights_to_train)
        p.set_gradients_as_graph_outputs = True
        self.training_parameters = p

        self.provider_options = provider_options
        self.sess_options = sess_options
        self.providers = providers
        self.run_options = run_options
        self.graph_builder_config = graph_builder_config

        # default
        if self.input_names is None:
            self.input_names = [obj.name
                                 for obj in self.onnx_model.graph.input]
        if self.output_names is None:
            self.output_names = [obj.name
                                 for obj in self.onnx_model.graph.output]
        if self.class_name is None:
            self.class_name = "TorchOrtFunction_%r" % id(self)
        if self.providers in (None, 'cpu'):
            self.providers = ["CPUExecutionProvider"]
            if self.provider_options is None:
                self.provider_options = [{}]
        if self.run_options is None:
            self.run_options = RunOptions()
            self.run_options.training_mode = True
        
        if self.graph_builder_config is None:
            initializer_names = [i.name for i in self.onnx_model.graph.initializer]
            input_names = [i.name for i in self.onnx_model.graph.input]
            
            config = OrtModuleGraphBuilderConfiguration()
            config.initializer_names = initializer_names
            config.initializer_names_to_train = list(
                sorted(self.training_parameters.weights_to_train))
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
    
    def create_class(self, enable_logging=False):
        """
        
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
            logger.info("[TorchOrtFactory] input_names=%r" % self.input_names)
            logger.info("[TorchOrtFactory] output_names=%r" % self.output_names)
            logger.info("[TorchOrtFactory] weights_to_train=%r" % self.training_parameters.weights_to_train)
            
        builder = OrtModuleGraphBuilder()
        if logger is not None:
            logger.info("[TorchOrtFactory] OrtModuleGraphBuilder.initialize")
        builder.initialize(
            self.onnx_model.SerializeToString(),
            self.graph_builder_config)
        if logger is not None:
            logger.info("[TorchOrtFactory] OrtModuleGraphBuilder.get_model")
        train_onnx_model = builder.get_model()
            
        if logger is not None:
            logger.info("[TorchOrtFactory] create TrainSession")
        sess = TrainingSession(
            train_onnx_model.SerializeToString(),
            parameters=self.training_parameters,
            provider_options=self.provider_options,
            sess_options=self.sess_options,
            providers=self.providers)
            
        if logger is not None:
            logger.info("[TorchOrtFactory] create training agent")
        training_agent = TrainingAgent(
            ort._sess,
            self.input_names,
            fw_outputs_device_info,
            bw_fetches_names,
            bw_outputs_device_info)

        if logger is not None:
            logger.info(
                "[TorchOrtFactory] instantiate dynamic class "
                "%r" % self.class_name)
            logger.info(
                "[TorchOrtFactory] input_names=%r" % self.input_names)
            logger.info(
                "[TorchOrtFactory] output_names=%r" % self.output_names)
            logger.info(
                "[TorchOrtFactory] weights_to_train=%r"
                "" % self.weights_to_train)

        newclass = type(
            self.class_name, (TorchOrtFunction,),
            {'__doc__': doc,
             '__module__': __name__,
             '_run_options': self.run_options,
             '_sess': sess,
             '_training_agent': training_agent,
             '_cache': OrtValueCache(),
             '_update_cache': False,
             '_state_': [],
             '_logger': logger,
             '_input_names': self.input_names,
             '_output_names': self.output_names,
             '_weights_to_train': list(sorted(self.weights_to_train)),
             'forward': staticmethod(ort_forward),
             'backward': staticmethod(ort_backward)})

        return newclass
    