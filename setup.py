#!/usr/bin/env python

import os
from setuptools import setup

README = os.path.join(os.getcwd(), "README.rst")


with open(README) as f:
    long_description = f.read()

setup(name='coremltools',
    version='0.6.3',
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
        'coremltools.models',
        'coremltools.proto',
        'coremltools._scripts'
    ],
    package_data={'': ['LICENSE.txt', 'README.rst', 'libcaffeconverter.so', 'libcoremlpython.so']},
    install_requires=[
        'numpy >= 1.6.2',
        'protobuf >= 3.1.0',
        'six==1.10.0'
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
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
          ],
    license='BSD'
)
