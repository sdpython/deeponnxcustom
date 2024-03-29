# -*- coding: utf-8 -*-
"""
Configuration for the documntation.
"""
import sys
import os
import warnings
import pydata_sphinx_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables

sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "deeponnxcustom", "Xavier Dupré", 2023,
                     "pydata_sphinx_theme", ['_static'],
                     locals(), extlinks=dict(issue=(
                         'https://github.com/sdpython/deeponnxcustom/issues/%s',
                         'issue %s')),
                     title="deeponnxcustom", book=True, branch='main')

extensions.extend([
    "sphinxcontrib.blockdiag"
])

blog_root = "http://www.xavierdupre.fr/app/deeponnxcustom/helpsphinx/"

html_css_files = ['my-styles.css']

html_logo = "_static/project_ico.png"
html_sidebars = {}
language = "en"

mathdef_link_only = True

custom_preamble = """\n
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\loinormale}[2]{{\\cal N}\\pa{#1,#2}}
\\newcommand{\\independant}[0]{\\;\\makebox[3ex]
{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]
{\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
\\newcommand{\\esp}{\\mathbb{E}}
\\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
\\newcommand{\\loi}[0]{{\\cal L}}
\\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\dans}[0]{\\rightarrow}
\\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
\\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
\\newcommand{\\loimultinomiale}[1]{{\\cal M}\\pa{#1}}
\\newcommand{\\trace}[1]{tr\\pa{#1}}
\\newcommand{\\abs}[1]{\\left|#1\\right|}
"""

# \\usepackage{eepic}
imgmath_latex_preamble += custom_preamble
latex_elements['preamble'] += custom_preamble

intersphinx_mapping.update({
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mlprodict':
        ('http://www.xavierdupre.fr/app/mlprodict/helpsphinx/', None),
    'onnxcustom':
        ('http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/', None),
    'pandas_streaming':
        ('http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
    'pyquickhelper':
        ('http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
})

epkg_dictionary.update({
    'autograd functions':
        'https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html',
    'C': 'https://en.wikipedia.org/wiki/C_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'cython': 'https://cython.org/',
    'DLPack': 'https://nnabla.readthedocs.io/en/latest/python/api/utils/dlpack.html',
    'DOT': 'https://www.graphviz.org/doc/info/lang.html',
    'ImageNet': 'http://www.image-net.org/',
    'InferenceSession':
        'https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.InferenceSession',
    'LightGBM': 'https://lightgbm.readthedocs.io/en/latest/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'mlprodict':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'NMF':
        'https://scikit-learn.org/stable/modules/generated/'
        'sklearn.decomposition.NMF.html',
    'numpy': 'https://numpy.org/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'onnxcustom': 'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/index.html',
    'ONNX operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX ML Operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'ONNX Zoo': 'https://github.com/onnx/models',
    'onnxmltools': 'https://github.com/onnx/onnxmltools',
    'OnnxPipeline':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/'
        'sklapi/onnx_pipeline.html?highlight=onnxpipeline',
    'onnxruntime': 'https://microsoft.github.io/onnxruntime/',
    'onnxruntime-training':
        'https://github.com/microsoft/onnxruntime/tree/master/orttraining',
    'openmp': 'https://en.wikipedia.org/wiki/OpenMP',
    'ORTModule': 'https://onnxruntime.ai/docs/',
    'OrtModuleGraphBuilderConfiguration':
        'http://www.xavierdupre.fr/app/deeponnxcustom/helpsphinx/glossary.html',
    'py-spy': 'https://github.com/benfred/py-spy',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'python': 'https://www.python.org/',
    'pytorch': 'https://pytorch.org/',
    'RunOptions':
        'https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.RunOptions',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'SessionOptions':
        'https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions',
    'skorch': 'https://skorch.readthedocs.io/en/stable/',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'sphinx-gallery': 'https://github.com/sphinx-gallery/sphinx-gallery',
    'Stochastic Gradient Descent':
        'https://en.wikipedia.org/wiki/Stochastic_gradient_descent',
    'torch': 'https://pytorch.org/',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'TrainingSession':
        'http://www.xavierdupre.fr/app/deeponnxcustom/helpsphinx/glossary.html',
    'TreeEnsembleRegressor':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md'
        '#ai.onnx.ml.TreeEnsembleRegressor',
    'xgboost': 'https://xgboost.readthedocs.io/en/latest/',
    'XGBoost': 'https://xgboost.readthedocs.io/en/latest/',
})

nblinks = {
    'alter_pipeline_for_debugging': 'http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/onnxcustom/helpers/pipeline.html#onnxcustom.helpers.pipeline.alter_pipeline_for_debugging',
}

warnings.filterwarnings("ignore", category=FutureWarning)
