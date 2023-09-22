//
//  ModelCreationUtils.h
//  CoreML_tests
//
//  Created by Anil Katti on 4/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "Format.hpp"
#include "Model.hpp"

#include "framework/TestUtils.hpp"

typedef struct {
    const char *name;
    int dimension;
} TensorAttributes;

CoreML::Specification::NeuralNetwork* buildBasicUpdatableNeuralNetworkModel(CoreML::Specification::Model& m);

CoreML::Specification::NeuralNetwork* buildBasicNeuralNetworkModel(CoreML::Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr, int numberOfLayers = 1, bool areWeightsQuantized = false, bool isBiasQuantized = false);

CoreML::Specification::NeuralNetworkClassifier* buildBasicNeuralNetworkClassifierModel(CoreML::Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, std::vector<std::string> stringClassLabels, std::vector<int64_t> intClassLabels, bool includeBias);

CoreML::Specification::KNearestNeighborsClassifier* buildBasicNearestNeighborClassifier(CoreML::Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const char *outTensorName);

CoreML::Specification::Pipeline* buildEmptyPipelineModel(CoreML::Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr);

CoreML::Specification::Pipeline* buildEmptyPipelineModelWithStringOutput(CoreML::Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const char *outTensorName);

void addCategoricalCrossEntropyLossWithSoftmaxAndSGDOptimizer(CoreML::Specification::Model& m, const char *softmaxInputName);

CoreML::Specification::NeuralNetwork* addInnerProductLayer(CoreML::Specification::Model& m, bool isUpdatable, const char *name, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr, bool areWeightsQuantized = false, bool isBiasQuantized = false);

CoreML::Specification::NeuralNetwork* addSoftmaxLayer(CoreML::Specification::Model& m, const char *name,  const char *input, const char *output);

void createSimpleNeuralNetworkClassifierModel(CoreML::Specification::Model *spec, const char *inputName, const char *outputName);

void createSimpleFeatureVectorizerModel(CoreML::Specification::Model *spec, const char *outputName, CoreML::Specification::ArrayFeatureType_ArrayDataType arrayType, int inputSize = 3);
