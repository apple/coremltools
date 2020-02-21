#!/usr/bin/env python
#
# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import imp
import os
from setuptools import setup

# Get the coremltools version string
coremltools_dir = os.path.join(os.path.dirname(__file__), 'coremltools')
version_module = imp.load_source('coremltools.version',
                                 os.path.join(coremltools_dir, 'version.py'))
__version__ = version_module.__version__

README = os.path.join(os.getcwd(), "README.rst")

with open(README) as f:
    long_description = f.read()

setup(name='coremltools',
      version=__version__,
      description='Community Tools for Core ML',
      long_description=long_description,
      author='Apple Inc.',
      author_email='coremltools@apple.com',
      url='https://github.com/apple/coremltools',
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
          'coremltools.converters.nnv2',
          'coremltools.converters.nnv2.backend',
          'coremltools.converters.nnv2.backend.nnv1',
          'coremltools.converters.nnv2.backend.nnv2',
          'coremltools.converters.nnv2.builtin_types',
          'coremltools.converters.nnv2.builtin_types.builtins',
          'coremltools.converters.nnv2.builtin_types.serialization',
          'coremltools.converters.nnv2.frontend',
          'coremltools.converters.nnv2.frontend.tensorflow',
          'coremltools.converters.nnv2.frontend.tensorflow.tf_graph_pass',
          'coremltools.converters.nnv2.nnv2_program',
          'coremltools.converters.nnv2.nnv2_program.ops',
          'coremltools.converters.nnv2.nnv2_program.passes',
          'coremltools.converters.nnv2.nnv2_program.program',
          'coremltools.graph_visualization',
          'coremltools.models',
          'coremltools.models.program',
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
          'numpy >= 1.14.5',
          'protobuf >= 3.1.0',
          'six>=1.10.0'
      ],
      entry_points={
          'console_scripts': ['coremlconverter = coremltools:_main']
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
      ],
      license='BSD'
      )
