
.. image:: https://circleci.com/gh/sdpython/deeponnxcustom/tree/main.svg?style=svg
    :target: https://circleci.com/gh/sdpython/deeponnxcustom/tree/main

.. image:: https://travis-ci.com/sdpython/deeponnxcustom.svg?branch=main
    :target: https://app.travis-ci.com/github/sdpython/deeponnxcustom
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/g9wt6riyh6n74t23?svg=true
    :target: https://ci.appveyor.com/project/sdpython/deeponnxcustom
    :alt: Build Status Windows

.. image:: https://codecov.io/gh/sdpython/deeponnxcustom/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/sdpython/deeponnxcustom

.. image:: https://badge.fury.io/py/deeponnxcustom.svg
    :target: http://badge.fury.io/py/deeponnxcustom

.. image:: http://img.shields.io/github/issues/sdpython/deeponnxcustom.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/deeponnxcustom/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://pepy.tech/badge/deeponnxcustom/month
    :target: https://pepy.tech/project/deeponnxcustom/month
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/deeponnxcustom.svg
    :target: https://github.com/sdpython/deeponnxcustom/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/deeponnxcustom.svg
    :target: https://github.com/sdpython/deeponnxcustom/
    :alt: Stars

.. image:: https://img.shields.io/github/repo-size/sdpython/deeponnxcustom
    :target: https://github.com/sdpython/deeponnxcustom/
    :alt: size

deeponnxcustom: custom ONNX and deep learning
=============================================

.. image:: https://raw.githubusercontent.com/sdpython/deeponnxcustom/main/doc/_static/logo.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/deeponnxcustom/helpsphinx/index.html>`_

Onnx, onnxruntime, deep learning, pytorch...

::

    python setup.py build_ext --inplace

Generate the setup in subfolder ``dist``:

::

    python setup.py sdist

Generate the documentation in folder ``dist/html``:

::

    python setup.py build_sphinx

Run the unit tests:

::

    python setup.py unittests

To check style:

::

    python -m flake8 deeponnxcustom tests examples

The function *check* or the command line ``python -m deeponnxcustom check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import deeponnxcustom
    deeponnxcustom.check()
