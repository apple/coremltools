//
//  ModelCreationUtils.mm
//  CoreML_tests
//
//  Created by Anil Katti on 4/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ModelCreationUtils.hpp"
#include "ParameterTests.hpp"

using namespace CoreML;

Specification::NeuralNetwork* buildBasicUpdatableNeuralNetworkModel(Specification::Model& m) {
    TensorAttributes inTensorAttr = { "A", 3 };
    TensorAttributes outTensorAttr = { "B", 1 };
    
    return buildBasicNeuralNetworkModel(m, true, &inTensorAttr, &outTensorAttr);
}

Specification::NeuralNetwork* buildBasicNeuralNetworkModel(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr, int numberOfLayers) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorAttr->name);
    auto outTensorShape = outTensor->mutable_type()->mutable_multiarraytype();
    outTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < outTensorAttr->dimension; i++) {
        outTensorShape->add_shape(1);
    }
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto neuralNet = m.mutable_neuralnetwork();
    
    for (int i = 0; i < numberOfLayers; i++) {
        auto layer = neuralNet->add_layers();
        
        std::string layerName = "inner_layer" + std::to_string(i);
        std::string input = "output" + std::to_string(i-1);
        std::string output = "output" + std::to_string(i);
        
        layer->set_name((numberOfLayers == 1) ? "inner_layer" : layerName.c_str());
        layer->add_input((i == 0) ? inTensorAttr->name : input.c_str());
        layer->add_output((i == numberOfLayers-1) ? outTensorAttr->name : output.c_str());
        
        Specification::InnerProductLayerParams *innerProductParams = layer->mutable_innerproduct();
        innerProductParams->set_inputchannels(1);
        innerProductParams->set_outputchannels(1);
        innerProductParams->mutable_weights()->add_floatvalue(1.0);
        innerProductParams->set_hasbias(true);
        innerProductParams->mutable_bias()->add_floatvalue(1.0);
        
        if (isUpdatable) {
            layer->set_isupdatable(true);
            innerProductParams->mutable_weights()->set_isupdatable(true);
            innerProductParams->mutable_bias()->set_isupdatable(true);
        }
    }
    
    if (isUpdatable) {
        m.set_isupdatable(true);
        
        for (auto feature : m.description().input()) {
            auto trainingInput = m.mutable_description()->mutable_traininginput()->Add();
            trainingInput->CopyFrom(feature);
        }
        for (auto feature : m.description().output()) {
            auto trainingInput = m.mutable_description()->mutable_traininginput()->Add();
            trainingInput->CopyFrom(feature);
        }
    }
    
    return neuralNet;
}

template <class NeuralNetworkType> void tempMethod0(NeuralNetworkType *nn) {
#pragma unused (nn)
}

Specification::NeuralNetworkClassifier* buildBasicNeuralNetworkClassifierModel(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, std::vector<std::string> classLabels) {
    
    auto input = m.mutable_description()->add_input();
    auto inputType = new Specification::FeatureType;
    auto multiArray = inputType->mutable_multiarraytype();
    multiArray->mutable_shape()->Add(inTensorAttr->dimension);
    multiArray->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_DOUBLE);
    input->set_name(inTensorAttr->name);
    input->set_allocated_type(inputType);
    
    auto output = m.mutable_description()->add_output();
    auto outputType = new Specification::FeatureType;
    outputType->mutable_stringtype();
    output->set_name("predictedClass");
    output->set_allocated_type(outputType);
    
    output = m.mutable_description()->add_output();
    outputType = new Specification::FeatureType;
    auto dictionary = outputType->mutable_dictionarytype();
    dictionary->mutable_stringkeytype();
    output->set_name("classProbabilities");
    output->set_allocated_type(outputType);

    m.mutable_description()->set_predictedfeaturename("predictedClass");
    m.mutable_description()->set_predictedprobabilitiesname("classProbabilities");
    
    auto classifier = m.mutable_neuralnetworkclassifier();
    
    for (auto className : classLabels) {
        classifier->mutable_stringclasslabels()->add_vector(className);
    }
    
    // Add inner product layer
    auto innerProductLayer = classifier->add_layers();
    innerProductLayer->set_name("inner_product");
    innerProductLayer->add_input(inTensorAttr->name);
    innerProductLayer->add_output("intermediateOutput");
    
    uint64_t C_in = (uint64_t)inTensorAttr->dimension;
    uint64_t C_out = (uint64_t)classLabels.size();
    
    auto innerProductParams = innerProductLayer->mutable_innerproduct();
    innerProductParams->set_inputchannels(C_in);
    innerProductParams->set_outputchannels(C_out);
    
    ::google::protobuf::RepeatedField<float>* weightsWrite = innerProductParams->mutable_weights()->mutable_floatvalue();
    weightsWrite->Resize((int)C_in * (int)C_out, 0.0);
    float *destination_weights = weightsWrite->mutable_data();
    for (uint64_t i = 0; i < C_in; i++) {
        for (uint64_t j = 0; j < C_out; j++) {
            float random = float(rand())/(RAND_MAX);
            destination_weights[i * C_out + j] = random;
        }
    }
    
    // Add inner product layer
    auto softmaxLayer = classifier->add_layers();
    softmaxLayer->set_name("softmax");
    softmaxLayer->add_input("intermediateOutput");
    softmaxLayer->add_output("scoreVector");
    softmaxLayer->mutable_softmax();
    
    if (isUpdatable) {
        m.set_isupdatable(true);
        innerProductLayer->set_isupdatable(true);
        innerProductParams->mutable_weights()->set_isupdatable(true);
        
        auto updateParams = classifier->mutable_updateparams();
        auto lossLayer = updateParams->add_losslayers();
        lossLayer->set_name("cross_entropy_loss");
        
        auto ceLossLayer = lossLayer->mutable_categoricalcrossentropylosslayer();
        ceLossLayer->set_input("scoreVector");
        ceLossLayer->set_target("target");
        
        addLearningRate(classifier, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
        addMiniBatchSize(classifier, Specification::Optimizer::kSgdOptimizer, 1, 1, 100, std::set<int64_t>());
        addEpochs(classifier, 100, 0, 100, std::set<int64_t>());
    }
    
    return classifier;
}

Specification::KNearestNeighborsClassifier* buildBasicNearestNeighborClassifier(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const char *outTensorName) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorName);
    outTensor->mutable_type()->mutable_stringtype();
    
    m.mutable_description()->set_predictedfeaturename(outTensorName);
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto nearestNeighborClassifier = m.mutable_knearestneighborsclassifier();
    nearestNeighborClassifier->set_k(3);
    
    auto nearestNeighborIndex = nearestNeighborClassifier->mutable_nearestneighborsindex();
    
    nearestNeighborIndex->mutable_singlekdtreeindex()->set_leafsize(30);
    nearestNeighborIndex->mutable_squaredeuclideandistance();
    
    nearestNeighborIndex->set_numberofdimensions(inTensorAttr->dimension);
    
    auto floatVector = nearestNeighborIndex->add_floatsamples();
    auto pointVector = floatVector->mutable_vector();
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        pointVector->Add((float)i);
    }
    
    nearestNeighborClassifier->mutable_uniformweighting();
    nearestNeighborClassifier->mutable_stringclasslabels()->add_vector(std::string("zero"));
    
    if (isUpdatable) {
        m.set_isupdatable(true);
    }
    
    return nearestNeighborClassifier;
}

Specification::Pipeline* buildEmptyPipelineModel(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorAttr->name);
    auto outTensorShape = outTensor->mutable_type()->mutable_multiarraytype();
    outTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < outTensorAttr->dimension; i++) {
        outTensorShape->add_shape(1);
    }
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto pipeline = m.mutable_pipeline();
    
    if (isUpdatable) {
        m.set_isupdatable(true);
    }
    
    return pipeline;
}

Specification::Pipeline* buildEmptyPipelineModelWithStringOutput(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const char *outTensorName) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorName);
    outTensor->mutable_type()->mutable_stringtype();
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto pipeline = m.mutable_pipeline();
    
    if (isUpdatable) {
        m.set_isupdatable(true);
    }
    
    return pipeline;
}

void addCategoricalCrossEntropyLossWithSoftmaxAndSGDOptimizer(Specification::Model& m, const char *softmaxInputName) {
    
    auto neuralNets = m.mutable_neuralnetwork();

    // set a softmax layer
    auto softmaxLayer = neuralNets->add_layers();
    softmaxLayer->set_name("softmax");
    softmaxLayer->add_input(softmaxInputName);
    softmaxLayer->add_output("softmax_out");
    softmaxLayer->mutable_softmax();

    Specification::NetworkUpdateParameters *updateParams = neuralNets->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("cross_entropy_loss_layer");
    Specification::CategoricalCrossEntropyLossLayer *ceLossLayer = lossLayer->mutable_categoricalcrossentropylosslayer();
    ceLossLayer->set_input("softmax_out");
    ceLossLayer->set_target("target");
    
    addLearningRate(neuralNets, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(neuralNets, Specification::Optimizer::kSgdOptimizer, 10, 5, 100, std::set<int64_t>());
    addEpochs(neuralNets, 100, 0, 100, std::set<int64_t>());
}


