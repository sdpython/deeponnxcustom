
Tutorial
========

.. index:: tutorial

The following examples explores the training of a deep or not
so deep machine learning model and the combination of :epkg:`pytorch`
and :epkg:`onnxruntime_training`.

.. toctree::
    :maxdepth: 2

    tutorial_training_ml
    benchmark

The tutorial was tested with following version:

.. runpython::
    :showcode:

    import sys
    import numpy
    import scipy
    import sklearn
    import lightgbm
    import onnx
    import onnxmltools
    import onnxruntime
    import xgboost
    import skl2onnx
    import mlprodict
    import onnxcustom
    import pyquickhelper

    print("python {}".format(sys.version_info))
    mods = [numpy, scipy, sklearn, lightgbm, xgboost,
            onnx, onnxmltools, onnxruntime, onnxcustom,
            skl2onnx, mlprodict, pyquickhelper]
    mods = [(m.__name__, m.__version__) for m in mods]
    mx = max(len(_[0]) for _ in mods) + 1
    for name, vers in sorted(mods):
        print("{}{}{}".format(name, " " * (mx - len(name)), vers))
