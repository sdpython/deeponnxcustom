
.. index:: glossary

Glossary
========

.. glossary::

    TrainingAgent
        This classes creates a training graph assuming the gradient
        is computed outside ONNX.
        See `training_agent.cc
        <https://github.com/microsoft/onnxruntime/blob/master/
        orttraining/orttraining/core/agent/training_agent.cc>`_.

    TrainingSession
        Extension of :epkg:`InferenceSession` to train a model ONNX.
        It assumes the input graph contains nodes computing
        a loss. The gradient is computed based on that.
        See `training_session.cc
        <https://github.com/microsoft/onnxruntime/blob/master/
        orttraining/orttraining/core/session/training_session.cc>`_.

    OrtModuleGraphBuilderConfiguration
        See `ortmodule_graph_builder.cc
        <https://github.com/microsoft/onnxruntime/blob/master/
        orttraining/orttraining/core/framework/ortmodule_graph_builder.cc>`_.
