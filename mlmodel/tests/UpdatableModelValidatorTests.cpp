//
//  UpdatableModelValidatorTests.cpp
//  CoreML_framework
//
//  Created by aseem wadhwa on 2/12/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "MLModelTests.hpp"
#include "../src/Format.hpp"
#include "../src/Model.hpp"
#include "../src/NeuralNetwork/NeuralNetworkShapes.hpp"
#include "ParameterTests.hpp"
#include "ModelCreationUtils.hpp"
#include "../src/Utils.hpp"

#include "framework/TestUtils.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

using namespace CoreML;


int testInvalidUpdatableModelWrongType() {
    
    /*
     checks that isUpdatable property is true only when the model type is
     - NN, KNN
     */
    
    Specification::Model m1;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *type_in = topIn->mutable_type()->mutable_multiarraytype();
    type_in->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    type_in->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("output");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    
    m1.mutable_identity();
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    Result res = Model::validate(m1);
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidUpdatableModelWrongLayer() {
    
    /*
     checks that isUpdatable property is true only for an updatable layer (conv or innerproduct)
     */
    
    Specification::Model m1;
    
    int num_inputs = 5;
    int num_outputs = 3;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    shape->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("probs");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);

    
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    const auto nn = m1.mutable_neuralnetwork();
    
    Specification::NeuralNetworkLayer *embeddingLayer = nn->add_layers();
    embeddingLayer->add_input("input");
    embeddingLayer->add_output("probs");
    embeddingLayer->set_isupdatable(true);
    embeddingLayer->set_name("embed");
    
    auto *params = embeddingLayer->mutable_embedding();
    params->set_inputdim(num_inputs);
    params->set_outputchannels(num_outputs);
    
    params->set_hasbias(true);
    
    for (int i = 0; i < num_inputs * num_outputs; i++) {
        params->mutable_weights()->add_floatvalue(1.0);
    }
    
    for (int i = 0; i < num_outputs; i++) {
        params->mutable_bias()->add_floatvalue(1.0);
    }
    
    Result res = Model::validate(m1);

    ML_ASSERT_BAD(res);
    return 0;
}


int testInvalidUpdatableModelWrongWeights() {
    
    /*
     checks that updatable peroperty is true for weights if the layer is marked as updatable.
     */
    
    Specification::Model m1;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    shape->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("probs");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    const auto nn = m1.mutable_neuralnetwork();
    
    Specification::NeuralNetworkLayer *innerProductLayer = nn->add_layers();
    innerProductLayer->add_input("input");
    innerProductLayer->add_output("probs");
    innerProductLayer->set_isupdatable(true);
     innerProductLayer->set_name("ip");

    Specification::InnerProductLayerParams *innerProductParams = innerProductLayer->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->mutable_weights()->set_isupdatable(false);
    
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    innerProductParams->mutable_bias()->set_isupdatable(true);

    Result res = Model::validate(m1);
    
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidUpdatableModelWrongBiases() {
    
    /*
     checks that updatable peroperty is true for biases if the layer is marked as updatable.
     */
    
    Specification::Model m1;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    shape->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("probs");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    const auto nn = m1.mutable_neuralnetwork();
    
    Specification::NeuralNetworkLayer *innerProductLayer = nn->add_layers();
    innerProductLayer->add_input("input");
    innerProductLayer->add_output("probs");
    innerProductLayer->set_isupdatable(true);
    
    Specification::InnerProductLayerParams *innerProductParams = innerProductLayer->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->mutable_weights()->set_isupdatable(true);
    
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    innerProductParams->mutable_bias()->set_isupdatable(false);
    
    Result res = Model::validate(m1);
    
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidUpdatableModelNonUpdatableLayers() {
    
    /*
     checks that updatable property is true for atleast one layer if a model is updatable.
     */
    
    Specification::Model m1;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    shape->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("probs");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    const auto nn = m1.mutable_neuralnetwork();
    
    Specification::NeuralNetworkLayer *innerProductLayer = nn->add_layers();
    
    innerProductLayer->add_input("input");
    innerProductLayer->add_output("probs");
    
    Specification::InnerProductLayerParams *innerProductParams = innerProductLayer->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    
    Result res = Model::validate(m1);
    
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidUpdatableModelwithCollidedLayerAndLossLayerNames() {
    
    /*
     checks that updatable model has no collison on names for model layers and loss layers
     */
    
    Specification::Model m1;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    shape->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("probs");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    const auto nn = m1.mutable_neuralnetwork();
    
    Specification::NetworkUpdateParameters *updateParams = nn->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("name1");
    
    Specification::NeuralNetworkLayer *innerProductLayer = nn->add_layers();
    innerProductLayer->add_input("input");
    innerProductLayer->add_output("probs");
    innerProductLayer->set_name("name1");
    innerProductLayer->set_isupdatable(true);
    
    Specification::InnerProductLayerParams *innerProductParams = innerProductLayer->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->mutable_weights()->set_isupdatable(true);
    
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    innerProductParams->mutable_bias()->set_isupdatable(true);

    Result res = Model::validate(m1);
    
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidModelUnsupportedLayersForBP() {
    
    /* checks if there are layers between updatable-marked layers and loss function that do not support back-propagation
     input ---> inner_product (U) ----> ABS (not supported for BP) ---> pooling --> output
     */
    
    
    Specification::Model m;
    auto *topIn = m.mutable_description()->add_input();
    topIn->set_name("A");
    topIn->mutable_type()->mutable_multiarraytype();
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    shape->add_shape(1);
    shape->add_shape(1);
    shape->add_shape(1);
    
    auto *out = m.mutable_description()->add_output();
    out->set_name("B");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    
    m.set_isupdatable(true);
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto *nn = m.mutable_neuralnetwork();
    auto *l1 = nn->add_layers();
    l1->set_name("inner_layer");
    l1->add_input("A");
    l1->add_output("ip");
    l1->set_isupdatable(true);
    Specification::InnerProductLayerParams *innerProductParams = l1->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->mutable_weights()->set_isupdatable(true);
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    innerProductParams->mutable_bias()->set_isupdatable(true);
    
    auto *l2 = nn->add_layers();
    l2->set_name("abs_layer");
    l2->add_input("ip");
    l2->add_output("abs_out");
    auto *elem = l2->mutable_unary();
    elem->set_type(Specification::UnaryFunctionLayerParams::ABS);
    
    auto *l3 = nn->add_layers();
    l3->set_name("pooling_layer");
    l3->add_input("abs_out");
    l3->add_output("B");
    auto *params = l3->mutable_pooling();
    params->set_type(::Specification::PoolingLayerParams::AVERAGE);
    params->set_globalpooling(true);
    params->mutable_valid();
    
    Specification::NetworkUpdateParameters *updateParams = nn->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("loss_layer");
    
    Specification::CategoricalCrossEntropyLossLayer *ceLossLayer = lossLayer->mutable_categoricalcrossentropylosslayer();
    ceLossLayer->set_input("B");
    ceLossLayer->set_target("label_target");
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidUpdatableLayerMissingBias() {
    
    /*
     an inner product layer marked as updatable must have bias parameter
     */
    
    Specification::Model m1;
    
    auto *topIn = m1.mutable_description()->add_input();
    topIn->set_name("input");
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    shape->add_shape(1);
    
    auto *out = m1.mutable_description()->add_output();
    out->set_name("probs");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_INT32);
    
    m1.set_isupdatable(true);
    m1.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    const auto nn = m1.mutable_neuralnetwork();
    
    Specification::NeuralNetworkLayer *innerProductLayer = nn->add_layers();
    innerProductLayer->set_name("ip");
    innerProductLayer->add_input("input");
    innerProductLayer->add_output("probs");
    innerProductLayer->set_isupdatable(true);
    
    Specification::InnerProductLayerParams *innerProductParams = innerProductLayer->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    
    innerProductParams->set_hasbias(false);
    
    Result res = Model::validate(m1);
    
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidCategoricalCrossEntropyLossLayerInputs() {
    
    /* at least one of the inputs of the loss layer must be produced within the inference network,
     otherwise the model is invalid.
     */
    
    Specification::Model m;
    auto *topIn = m.mutable_description()->add_input();
    topIn->set_name("A");
    topIn->mutable_type()->mutable_multiarraytype();
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    shape->add_shape(1);
    shape->add_shape(1);
    shape->add_shape(1);
    
    auto *out = m.mutable_description()->add_output();
    out->set_name("B");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    
    m.set_isupdatable(true);
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto *nn = m.mutable_neuralnetwork();
    auto *l1 = nn->add_layers();
    l1->set_name("inner_layer");
    l1->add_input("A");
    l1->add_output("B");
    l1->set_isupdatable(true);
    Specification::InnerProductLayerParams *innerProductParams = l1->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->mutable_weights()->set_isupdatable(true);
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    innerProductParams->mutable_bias()->set_isupdatable(true);
    
    Specification::NetworkUpdateParameters *updateParams = nn->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("cross_entropy_loss_layer");
    
    Specification::CategoricalCrossEntropyLossLayer *ceLossLayer = lossLayer->mutable_categoricalcrossentropylosslayer();
    ceLossLayer->set_input("C");
    ceLossLayer->set_target("label_target");
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testInvalidMeanSquaredErrorLossLayerInputs() {
    
    /* at least one of the inputs of the loss layer must be produced within the inference network,
     otherwise the model is invalid.
     */
    
    Specification::Model m;
    auto *topIn = m.mutable_description()->add_input();
    topIn->set_name("A");
    topIn->mutable_type()->mutable_multiarraytype();
    auto *shape = topIn->mutable_type()->mutable_multiarraytype();
    shape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    shape->add_shape(1);
    shape->add_shape(1);
    shape->add_shape(1);
    
    auto *out = m.mutable_description()->add_output();
    out->set_name("B");
    auto *type_out = out->mutable_type()->mutable_multiarraytype();
    type_out->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    
    m.set_isupdatable(true);
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto *nn = m.mutable_neuralnetwork();
    auto *l1 = nn->add_layers();
    l1->set_name("inner_layer");
    l1->add_input("A");
    l1->add_output("B");
    l1->set_isupdatable(true);
    Specification::InnerProductLayerParams *innerProductParams = l1->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->mutable_weights()->set_isupdatable(true);
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    innerProductParams->mutable_bias()->set_isupdatable(true);
    
    Specification::NetworkUpdateParameters *updateParams = nn->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("mse_loss_layer");
    
    Specification::MeanSquaredErrorLossLayer *mseLossLayer = lossLayer->mutable_meansquarederrorlosslayer();
    mseLossLayer->set_input("C");
    mseLossLayer->set_target("label_target");
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

static Specification::NetworkUpdateParameters* buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(Specification::Model& m) {
    
    auto *nn = buildBasicUpdatableNeuralNetworkModel(m);

    // set a softmax layer
    auto softmaxLayer = nn->add_layers();
    softmaxLayer->set_name("softmax");
    softmaxLayer->add_input("B");
    softmaxLayer->add_output("softmax_out");
    softmaxLayer->mutable_softmax();
    
    Specification::NetworkUpdateParameters *updateParams = nn->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("cross_entropy_loss_layer");

    Specification::CategoricalCrossEntropyLossLayer *ceLossLayer = lossLayer->mutable_categoricalcrossentropylosslayer();
    ceLossLayer->set_input("softmax_out");
    ceLossLayer->set_target("label_target");
    
    return updateParams;
}

static Specification::NetworkUpdateParameters* buildBasicUpdatableModelWithMeanSquaredError(Specification::Model& m) {
    
    auto *nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    Specification::NetworkUpdateParameters *updateParams = nn->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("mean_squared_error_loss_layer");
    
    Specification::MeanSquaredErrorLossLayer *mseLossLayer = lossLayer->mutable_meansquarederrorlosslayer();
    mseLossLayer->set_input("B");
    mseLossLayer->set_target("label_target");
    
    return updateParams;
}

int testMissingUpdatableModelParameters() {
    
    Specification::Model m;
    
    // basic neural network model with CCE and softmax without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    
    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    // clear model and redo test with MSE
    m.Clear();
    
    // basic neural network model with MSE without any updatable model parameters.
    (void)buildBasicUpdatableModelWithMeanSquaredError(m);
    
    // expect validation to fail due to missing updatable model parameters.
    res = Model::validate(m);
    ML_ASSERT_BAD(res);

    return 0;
}

int testMissingMiniBatchSizeParameter() {
    
    Specification::Model m;
    
    // basic neural network model with CCE and softmax without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    
    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    // now add an updatable model parameter.
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    
    // expect validation to still fail due to missing mini batch size.
    res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 10, 5, 100, std::set<int64_t>());
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
    
    // expect validation to pass.
    res = Model::validate(m);
    ML_ASSERT_GOOD(res);
    return 0;
}

int testMissingLearningRateParameter() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    
    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    // now add an updatable model parameter.
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 10, 5, 100, std::set<int64_t>());
    
    // expect validation to still fail due to missing optimizer.
    res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
    
    // expect validation to pass.
    res = Model::validate(m);
    ML_ASSERT_GOOD(res);
    return 0;
}

int testMissingBeta1Parameter() {

    Specification::Model m;

    // basic neural network model without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);

    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);

    // now add an updatable model parameter.
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 10, 5, 100, std::set<int64_t>());
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
    addBeta2(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    addEps(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);

    res = Model::validate(m);
    ML_ASSERT_BAD(res);

    addBeta1(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);

    // expect validation to pass.
    res = Model::validate(m);
    ML_ASSERT_GOOD(res);
    return 0;
}

int testMissingBeta2Parameter() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    
    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    // now add an updatable model parameter.
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 10, 5, 100, std::set<int64_t>());
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
    addBeta1(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    addEps(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    
    res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    addBeta2(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    
    // expect validation to pass.
    res = Model::validate(m);
    ML_ASSERT_GOOD(res);
    return 0;
}

int testMissingEpsParameter() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    
    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    // now add an updatable model parameter.
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 10, 5, 100, std::set<int64_t>());
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
    addBeta1(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    addBeta2(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    
    res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    addEps(m.mutable_neuralnetwork(), Specification::Optimizer::kAdamOptimizer, 0.7f, 0.0f, 1.0f);
    
    // expect validation to pass.
    res = Model::validate(m);
    ML_ASSERT_GOOD(res);
    return 0;
}

int testMissingEpochsParameter() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    
    // expect validation to fail due to missing updatable model parameters.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    
    // now add an updatable model parameter.
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 10, 5, 100, std::set<int64_t>());
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    
    // expect validation to pass.
    res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testExistingShuffleWithMissingSeedParameter() {

    Specification::Model m;

    // basic neural network model without any updatable model parameters.
    (void)buildBasicUpdatableModelWithCategoricalCrossEntropyAndSoftmax(m);
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 10, 5, 100, std::set<int64_t>());
    addLearningRate(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());

    addShuffleAndSeed(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
    Result res = Model::validate(m);
    ML_ASSERT_GOOD(res);

    return 0;
}

int testNonUpdatablePipelineWithNonUpdatableModels() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline = buildEmptyPipelineModelWithStringOutput(spec, false, &tensorAttributesA, "E");
    
    auto m1 = pipeline->add_models();
    auto m2 = pipeline->add_models();
    auto m3 = pipeline->add_models();
    auto m4 = pipeline->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, false, &tensorAttributesA, &tensorAttributesB);
    (void)buildBasicNeuralNetworkModel(*m2, false, &tensorAttributesB, &tensorAttributesC);
    (void)buildBasicNeuralNetworkModel(*m3, false, &tensorAttributesC, &tensorAttributesD);
    (void)buildBasicNearestNeighborClassifier(*m4, false, &tensorAttributesD, "E");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    // expect validation to pass!
    res = Model::validate(spec);
    ML_ASSERT_GOOD(res);
    
    return 0;
}

int testNonUpdatablePipelineWithOneUpdatableModel() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline = buildEmptyPipelineModelWithStringOutput(spec, false, &tensorAttributesA, "E");
    
    auto m1 = pipeline->add_models();
    auto m2 = pipeline->add_models();
    auto m3 = pipeline->add_models();
    auto m4 = pipeline->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, false, &tensorAttributesA, &tensorAttributesB);
    (void)buildBasicNeuralNetworkModel(*m2, false, &tensorAttributesB, &tensorAttributesC);
    (void)buildBasicNeuralNetworkModel(*m3, false, &tensorAttributesC, &tensorAttributesD);
    (void)buildBasicNearestNeighborClassifier(*m4, true, &tensorAttributesD, "E");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    // expect validation to fail due to presence of uptable model in non-updatable pipeline.
    res = Model::validate(spec);
    ML_ASSERT_BAD(res);
    
    return 0;
}

int testNonUpdatablePipelineWithOneUpdatableModelInsidePipelineHierarchy() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline1 = buildEmptyPipelineModelWithStringOutput(spec, false, &tensorAttributesA, "E");
    
    auto m1 = pipeline1->add_models();
    auto m2 = pipeline1->add_models();
    auto m3 = pipeline1->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, false, &tensorAttributesA, &tensorAttributesB);
    auto pipeline2 = buildEmptyPipelineModel(*m2, false, &tensorAttributesB, &tensorAttributesD);
    (void)buildBasicNearestNeighborClassifier(*m3, false, &tensorAttributesD, "E");
    
    auto m4 = pipeline2->add_models();
    auto m5 = pipeline2->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m4, false, &tensorAttributesB, &tensorAttributesC);
    (void)buildBasicNeuralNetworkModel(*m5, true, &tensorAttributesC, &tensorAttributesD);
    addCategoricalCrossEntropyLossWithSoftmaxAndSGDOptimizer(*m5, "D");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_BAD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m5);
    ML_ASSERT_GOOD(res);
    
    // expect validation to fail due to presence of uptable model in non-updatable pipeline.
    res = Model::validate(spec);
    ML_ASSERT_BAD(res);
    
    return 0;
}

int testUpdatablePipelineWithNonUpdatableModels() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline = buildEmptyPipelineModelWithStringOutput(spec, true, &tensorAttributesA, "E");
    
    auto m1 = pipeline->add_models();
    auto m2 = pipeline->add_models();
    auto m3 = pipeline->add_models();
    auto m4 = pipeline->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, false, &tensorAttributesA, &tensorAttributesB);
    (void)buildBasicNeuralNetworkModel(*m2, false, &tensorAttributesB, &tensorAttributesC);
    (void)buildBasicNeuralNetworkModel(*m3, false, &tensorAttributesC, &tensorAttributesD);
    (void)buildBasicNearestNeighborClassifier(*m4, false, &tensorAttributesD, "E");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    // expect validation to fail due to missing updatable model in pipeline.
    res = Model::validate(spec);
    ML_ASSERT_BAD(res);
    
    return 0;
}

int testUpdatablePipelineWithMultipleUpdatableModels() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline = buildEmptyPipelineModelWithStringOutput(spec, true, &tensorAttributesA, "E");
    
    auto m1 = pipeline->add_models();
    auto m2 = pipeline->add_models();
    auto m3 = pipeline->add_models();
    auto m4 = pipeline->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, true, &tensorAttributesA, &tensorAttributesB);
    addCategoricalCrossEntropyLossWithSoftmaxAndSGDOptimizer(*m1, "B");
    
    (void)buildBasicNeuralNetworkModel(*m2, false, &tensorAttributesB, &tensorAttributesC);
    
    (void)buildBasicNeuralNetworkModel(*m3, true, &tensorAttributesC, &tensorAttributesD);
    addCategoricalCrossEntropyLossWithSoftmaxAndSGDOptimizer(*m3, "D");
    
    (void)buildBasicNearestNeighborClassifier(*m4, false, &tensorAttributesD, "E");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    // expect validation to fail due to multiple updatable models in the pipeline.
    res = Model::validate(spec);
    ML_ASSERT_BAD(res);
    
    return 0;
}

int testUpdatablePipelineWithOneUpdatableModel() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline = buildEmptyPipelineModelWithStringOutput(spec, true, &tensorAttributesA, "E");
    
    auto m1 = pipeline->add_models();
    auto m2 = pipeline->add_models();
    auto m3 = pipeline->add_models();
    auto m4 = pipeline->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, false, &tensorAttributesA, &tensorAttributesB);
    (void)buildBasicNeuralNetworkModel(*m2, false, &tensorAttributesB, &tensorAttributesC);
    (void)buildBasicNeuralNetworkModel(*m3, false, &tensorAttributesC, &tensorAttributesD);
    (void)buildBasicNearestNeighborClassifier(*m4, true, &tensorAttributesD, "E");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    // expect validation to pass!
    res = Model::validate(spec);
    ML_ASSERT_GOOD(res);
    
    return 0;
}

int testUpdatablePipelineWithOneUpdatableModelInsidePipelineHierarchy() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesA = { "A", 3 };
    TensorAttributes tensorAttributesB = { "B", 1 };
    TensorAttributes tensorAttributesC = { "C", 1 };
    TensorAttributes tensorAttributesD = { "D", 3 };
    
    auto pipeline1 = buildEmptyPipelineModelWithStringOutput(spec, true, &tensorAttributesA, "E");
    
    auto m1 = pipeline1->add_models();
    auto m2 = pipeline1->add_models();
    auto m3 = pipeline1->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m1, false, &tensorAttributesA, &tensorAttributesB);
    (void)buildBasicNeuralNetworkModel(*m2, false, &tensorAttributesB, &tensorAttributesC);
    auto pipeline2 = buildEmptyPipelineModelWithStringOutput(*m3, true, &tensorAttributesC, "E");
    
    auto m4 = pipeline2->add_models();
    auto m5 = pipeline2->add_models();
    
    (void)buildBasicNeuralNetworkModel(*m4, false, &tensorAttributesC, &tensorAttributesD);
    (void)buildBasicNearestNeighborClassifier(*m5, true, &tensorAttributesD, "E");
    
    res = Model::validate(*m1);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m2);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m3);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m4);
    ML_ASSERT_GOOD(res);
    
    res = Model::validate(*m5);
    ML_ASSERT_GOOD(res);
    
    // expect validation to pass!
    res = Model::validate(spec);
    ML_ASSERT_GOOD(res);
    
    return 0;
}

int testValidUpdatableModelWith1024Layers() {
    
    Specification::Model spec;
    Result res;
    TensorAttributes tensorAttributesIn = { "InTensor", 3 };
    TensorAttributes tensorAttributesOut = { "OutTensor", 1 };
    
    (void)buildBasicNeuralNetworkModel(spec, true, &tensorAttributesIn, &tensorAttributesOut, 1024);
    addCategoricalCrossEntropyLossWithSoftmaxAndSGDOptimizer(spec, "OutTensor");
    
    res = Model::validate(spec);
    ML_ASSERT_GOOD(res);
    
    return 0;
}

