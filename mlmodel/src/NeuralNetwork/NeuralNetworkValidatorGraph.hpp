//
//  NeuralNetworkValidatorGraph.h
//  mlmodel
//

#include "../Validators.hpp"

using namespace CoreML;

static bool isLayerSupportedForBackprop(const Specification::NeuralNetworkLayer* layer) {
    switch (layer->layer_case()) {
        case Specification::NeuralNetworkLayer::kConvolution:
        case Specification::NeuralNetworkLayer::kInnerProduct:
        case Specification::NeuralNetworkLayer::kSoftmax:
        case Specification::NeuralNetworkLayer::kPooling:
        case Specification::NeuralNetworkLayer::kBatchnorm:
            return true;
        case Specification::NeuralNetworkLayer::kActivation:{
            switch(layer->activation().NonlinearityType_case()) {
                case Specification::ActivationParams::NonlinearityTypeCase::kReLU:
                case Specification::ActivationParams::NonlinearityTypeCase::kSigmoid:
                case Specification::ActivationParams::NonlinearityTypeCase::kTanh:
                    return true;
                default:
                    return false;
            }
        }
        default:
            return false;
    }
}

struct node {
    
    public:
    
    std::vector<node> parents; // list of nodes that are parent to this node
    std::vector<node> children;
    Specification::NeuralNetworkLayer::LayerCase layerType;
    std::string name; // name of this node
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    bool isUpdatable;
    bool isBackPropagable;
    
    node () {}
    
    node(const Specification::LossLayer *lossLayer)
    {
        name = lossLayer->name();
        
        switch (lossLayer->LossLayerType_case()) {
            case CoreML::Specification::LossLayer::kCategoricalCrossEntropyLossLayer:
                inputNames.push_back(lossLayer->categoricalcrossentropylosslayer().input());
                break;
            case CoreML::Specification::LossLayer::kMeanSquaredErrorLossLayer:
                inputNames.push_back(lossLayer->meansquarederrorlosslayer().input());
                break;
            default:
                break;
        }

        isUpdatable = false;
        isBackPropagable = false;
    }
    
    node(const Specification::NeuralNetworkLayer* layer)
    {
        std::vector<std::string> inNames;
        std::vector<std::string> outNames;
        for (const auto& elem: layer->input()) {
            inNames.push_back(elem);
        }
        for (const auto& elem: layer->output()) {
            outNames.push_back(elem);
        }
        layerType =  layer->layer_case();
        inputNames = inNames;
        outputNames = outNames;
        name = layer->name();
        isUpdatable = layer->isupdatable();
        isBackPropagable = isLayerSupportedForBackprop(layer);
    }
};

struct NeuralNetworkValidatorGraph {
    
    public:
    
    std::map<std::string, node> nodeNameToNode;
    std::map<std::string, node> blobNameToProducingNode;
    
    NeuralNetworkValidatorGraph() {}
    
    void insertNode(node thisNode)
    {
        for (const auto& name: thisNode.inputNames) {
            if (blobNameToProducingNode.find(name) != blobNameToProducingNode.end()) {
                thisNode.parents.push_back(blobNameToProducingNode[name]);
                blobNameToProducingNode[name].children.push_back(thisNode);
            }
        }
        for (const auto& name: thisNode.outputNames) {
            blobNameToProducingNode[name] = thisNode;
        }
        nodeNameToNode[thisNode.name] = thisNode;
    }
    
    node getNodeFromName(std::string name) const
    {
        if (nodeNameToNode.find(name) == nodeNameToNode.end()) {
            return node();
        }
        return nodeNameToNode.at(name);
    }
};



