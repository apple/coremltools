#!/usr/bin/env python

import os
from setuptools import setup

README = os.path.join(os.getcwd(), "README.rst")


with open(README) as f:
    long_description = f.read()

setup(name='coremltools',
    version='3.0b1',
    description='Community Tools for CoreML',
    long_description=long_description,
    author='Apple Inc.',
    author_email='coremltools@apple.com',
    url='',
    packages=[
        'coremltools',
        'coremltools._deps',
        'coremltools.converters',
        'coremltools.converters.caffe',
        'coremltools.converters.sklearn',
        'coremltools.converters.xgboost',
        'coremltools.converters.libsvm',
        'coremltools.converters.keras',
        'coremltools.converters.tensorflow',
        'coremltools.converters.nnssa',
        'coremltools.converters.nnssa.commons',
        'coremltools.converters.nnssa.commons.builtins',
        'coremltools.converters.nnssa.commons.serialization',
        'coremltools.converters.nnssa.coreml',
        'coremltools.converters.nnssa.coreml.graph_pass',
        'coremltools.converters.nnssa.frontend',
        'coremltools.converters.nnssa.frontend.graph_pass',
        'coremltools.converters.nnssa.frontend.tensorflow',
        'coremltools.converters.nnssa.frontend.tensorflow.graph_pass',
        'coremltools.graph_visualization',
        'coremltools.models',
        'coremltools.models.neural_network',
        'coremltools.models.nearest_neighbors',
        'coremltools.proto',
        'coremltools._scripts'
    ],
    package_data={'': ['LICENSE.txt', 'README.rst', 'libcaffeconverter.so', 'libcoremlpython.so'],
                  'coremltools': ['graph_visualization/__init__.py',
                                  'graph_visualization/app.js',
                                  'graph_visualization/index.html',
                                  'graph_visualization/style.css',
                                  'graph_visualization/assets/*',
                                  'graph_visualization/icons/*']
                  },
    install_requires=[
        'numpy >= 1.10.0',
        'protobuf >= 3.1.0',
        'six>=1.10.0'
    ],
    entry_points = {
        'console_scripts': ['coremlconverter = coremltools:_main']
    },
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Operating System :: MacOS :: MacOS X',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
          ],
    license='BSD'
)
