//
//  UpdatableModelValidatorTests.cpp
//  CoreML_framework
//
//  Created by aseem wadhwa on 2/12/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "Format.hpp"
#include "Model.hpp"

#include "framework/TestUtils.hpp"

template <class NeuralNetworkClass> void addLearningRate(NeuralNetworkClass *nn, CoreML::Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue,  double maxValue);

template <class NeuralNetworkClass> void addMiniBatchSize(NeuralNetworkClass *nn, CoreML::Specification::Optimizer::OptimizerTypeCase optimizerType, int64_t defaultValue, int64_t minValue, int64_t maxValue, std::set<int64_t> allowedValues = std::set<int64_t>());

template <class NeuralNetworkClass> void addEpochs(NeuralNetworkClass *nn, int64_t defaultValue, int64_t minValue, int64_t maxValue, std::set<int64_t> allowedValues = std::set<int64_t>());

template <class NeuralNetworkClass> void addShuffleAndSeed(NeuralNetworkClass *nn, int64_t defaultValue, int64_t minValue, int64_t maxValue, std::set<int64_t> allowedValues);

template <class NeuralNetworkClass> void addCategoricalCrossEntropyLoss(CoreML::Specification::Model& m, NeuralNetworkClass *nn, const char *lossName, const char *softmaxInputName, const char *targetName);

template <class NeuralNetworkClass> void addMeanSquareError(CoreML::Specification::Model& m, NeuralNetworkClass *nn, const char *lossName, const char *mseInputName, const char *targetName);

void addMomentum(CoreML::Specification::NeuralNetwork *nn, CoreML::Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue);

void addBeta1(CoreML::Specification::NeuralNetwork *nn, CoreML::Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue);

void addBeta2(CoreML::Specification::NeuralNetwork *nn, CoreML::Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue);

void addEps(CoreML::Specification::NeuralNetwork *nn, CoreML::Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue);
