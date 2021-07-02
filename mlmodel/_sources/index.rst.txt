.. Core ML Format Reference documentation master file, created by
   sphinx-quickstart on Mon Jun 28 11:26:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################################
Core ML Model Format Specification
##################################

This document contains the protobuf message definitions
that comprise the Core ML model document (``.mlmodel``) format. The top-level
message is `Model`, which is defined in :file:`Model.proto`.
Other message types describe data structures, feature types,
feature engineering model types, and predictive model types.

.. toctree::
    :maxdepth: 1
    
    Format/Model.rst
    Format/NeuralNetwork.rst
    Format/MIL.rst
    Format/Regressors.rst
    Format/Classifiers.rst
    Format/OtherModels.rst
    Format/FeatureEngineering.rst
    Format/Pipeline.rst
    Format/Identity.rst
    Format/SVM.rst
    Format/DataStructuresTypes.rst

.. toctree::
   :maxdepth: 1
   :caption: Resources
   
   Guides and examples <https://coremltools.readme.io/docs>
   Core ML Tools (coremltools) API <https://apple.github.io/coremltools/index.html>
   GitHub <https://github.com/apple/coremltools/tree/main/mlmodel>
