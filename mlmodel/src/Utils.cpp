//
//  Utils.cpp
//  mlmodelspec
//
//  Created by Bill March on 10/3/17.
//  Copyright © 2017 Apple. All rights reserved.
//

#include "Utils.hpp"

using namespace CoreML;

static bool isCustomModel(const Specification::Model& model);
static bool hasFlexibleShapes(const Specification::Model& model);
static bool isAppleWordTagger(const Specification::Model& model);
static bool isAppleTextClassifier(const Specification::Model& model);
static bool isAppleGazetteer(const Specification::Model& model);
static bool isAppleWordEmbedding(const Specification::Model& model);
static bool isScenePrint(const Specification::Model& model);
static bool isObjectPrint(const Specification::Model& model);
static bool isAppleAudioFeatureExtractor(const Specification::Model& model);
static bool isSoundPrint(const Specification::Model& model);
static bool hasCategoricalSequences(const Specification::Model& model);
static bool isNonMaxSuppression(const Specification::Model& model);
static bool isBayesianProbitRegressor(const Specification::Model& model);
static bool hasIOS12NewNeuralNetworkLayers(const Specification::Model& model);
static bool hasIOS13NeuralNetworkFeatures(const Specification::Model& model);
static bool hasIOS14NeuralNetworkFeatures(const Specification::Model& model);
static bool hasDefaultValueForOptionalInputs(const Specification::Model& model);
static bool nonMaxSuppressionUsesFloat32InputsOrOutputs(const Specification::Model& model);
static bool hasFloat16MultiArray(const Specification::Model& model);
static bool hasGrayscaleFloat16Image(const Specification::Model& model);
static bool hasCoreML6Opsets(const Specification::Model& model);
static bool hasCoreML7Opsets(const Specification::Model& model);
static bool hasCoreML8Opsets(const Specification::Model& model);
static bool hasMultiFunctions(const Specification::Model& model);
static bool hasEmptyInput(const Specification::Model& model);
static bool hasWeightOfType(const Specification::Model& model, const WeightParamType& wt);

// Returning a pointer here because of verification issues with allocating this type on the stack
google::protobuf::RepeatedPtrField<Specification::NeuralNetworkLayer> const *CoreML::getNNSpec(const Specification::Model& model)  {
    switch (model.Type_case()) {
        case Specification::Model::TypeCase::kNeuralNetwork:
            return &(model.neuralnetwork().layers());
        case Specification::Model::TypeCase::kNeuralNetworkRegressor:
            return &(model.neuralnetworkregressor().layers());
        case Specification::Model::TypeCase::kNeuralNetworkClassifier:
            return &(model.neuralnetworkclassifier().layers());
        default:
//            assert(false);
            // Don't freak out about new, we don't really get here
            return NULL;
    }
}

/// A utility function to traverse a pipeline model and invoke a handler.
///
/// If the model is not a pipeline, it simply invokes the handler with the model.
///
/// If the model is a pipeline, it invokes the handler with each sub
/// model first, and then with the pipeline model.
///
/// The traversal stops if the handler returns `true`.
static bool walkModelAndPipeline(const Specification::Model &model,
                                 std::function<bool(const Specification::Model& m)> handler) {
    switch (model.Type_case()) {
        case Specification::Model::kPipeline:
            for (auto &m : model.pipeline().models()) {
                if (walkModelAndPipeline(m, handler)) {
                    return true;
                }
            }
            break;
        case Specification::Model::kPipelineRegressor:
            for (auto &m : model.pipelineregressor().pipeline().models()) {
                if (walkModelAndPipeline(m, handler)) {
                    return true;
                }
            }
            break;
        case Specification::Model::kPipelineClassifier:
            for (auto &m : model.pipelineclassifier().pipeline().models()) {
                if (walkModelAndPipeline(m, handler)) {
                    return true;
                }
            }
            break;
        default:
            break;
    }
    if (handler(model)) {
        return true;
    }
    return false;
}

/// A utility function to invoke a handler with input, output, and
/// state feature description.
///
/// The `description` parameter template can be either a model
/// description or a function description.
///
/// The function returns `true` if any of the feature descriptions met
/// the specified criteria predicate function.
template <typename Description>
static bool anyFeatureDescriptionOfDescription(const Description &description,
                                               std::function<bool(const Specification::FeatureDescription& fd)> criteria) {
    for (const auto& fd: description.input()) {
        if (criteria(fd)) {
            return true;
        }
    }

    for (const auto& fd: description.output()) {
        if (criteria(fd)) {
            return true;
        }
    }

    for (const auto& fd: description.state()) {
        if (criteria(fd)) {
            return true;
        }
    }
    return false;
}

/// A utility function to invoke a handler with input, output, and
/// state feature description of the model.
///
/// It traverses a pipeline model.
///
/// The function returns `true` if any of the feature descriptions met
/// the specified criteria predicate function.
static bool anyFeatureDescriptionOfModel(const Specification::Model &model,
                                         std::function<bool(const Specification::FeatureDescription& fd)> criteria) {
    return walkModelAndPipeline(
        model,
        [&criteria](const Specification::Model& m) {
            const auto& modelDescription = m.description();
            if (modelDescription.functions_size() == 0) {
                // A single function syntax
                if (anyFeatureDescriptionOfDescription(modelDescription, criteria)) {
                    return true;
                }
            } else {
                // A multi function syntax
                for (const auto& functionDescription: modelDescription.functions()) {
                    if (anyFeatureDescriptionOfDescription(modelDescription, criteria)) {
                        return true;
                    }
                }
            }
            return false;
        });
}

bool CoreML::hasCustomLayer(const Specification::Model& model) {
    return walkModelAndPipeline(
        model,
        [](const auto& m) {
            if (const auto *layers = getNNSpec(m)) {
                for (const auto& layer: *layers) {
                    if (layer.layer_case() == Specification::NeuralNetworkLayer::kCustom) {
                        return true;
                    }
                }
            }
            return false;
        });
}

std::vector<StringPair> CoreML::getCustomLayerNamesAndDescriptions(const Specification::Model& model) {
    std::vector<std::pair<std::string, std::string> > retval;
    walkModelAndPipeline(model, [&retval](const auto& m) {
        if (const auto *layers = getNNSpec(m)) {
            for (const auto& layer: *layers) {
                if (layer.layer_case() == Specification::NeuralNetworkLayer::kCustom) {
                    retval.push_back(std::make_pair(layer.custom().classname(), layer.custom().description()));
                }
            }
        }
        return false; // Do not stop walking
    });
    return retval;
}

std::vector<std::pair<std::string, std::string> > CoreML::getCustomModelNamesAndDescriptions(const Specification::Model& model) {
    std::vector<std::pair<std::string, std::string> > retval;
    walkModelAndPipeline(model, [&retval](const auto& m) {
        if (m.Type_case() == Specification::Model::kCustomModel) {
            retval.push_back(std::make_pair(m.custommodel().classname(), m.custommodel().description()));
        }
        return false; // Do not stop walking
    });
    return retval;
}

void CoreML::downgradeSpecificationVersion(Specification::Model *pModel) {

    if (!pModel) { return; }

    if (pModel->specificationversion() == 0 || pModel->specificationversion() > MLMODEL_SPECIFICATION_VERSION_NEWEST) {
        // If mistakenly set specification version or never set and left as default
        // lets start at the newest specification version and downgrade from there
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_NEWEST);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS18 && !hasIOS18Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS17);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS17 && !hasIOS17Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS16);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS16 && !hasIOS16Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS15);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS15 && !hasIOS15Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS14);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS14 && !hasIOS14Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS13 && !hasIOS13Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS12);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS12 && !hasIOS12Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS11_2);
    }

    if (pModel->specificationversion() == MLMODEL_SPECIFICATION_VERSION_IOS11_2 && !hasIOS11_2Features(*pModel)) {
        pModel->set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS11);
    }

    ::CoreML::Specification::Pipeline *pipeline = NULL;
    auto modelType = pModel->Type_case();
    if (modelType == Specification::Model::kPipeline) {
        pipeline = pModel->mutable_pipeline();
    } else if (modelType == Specification::Model::kPipelineRegressor) {
        pipeline = pModel->mutable_pipelineregressor()->mutable_pipeline();
    } else if (modelType == Specification::Model::kPipelineClassifier) {
        pipeline = pModel->mutable_pipelineclassifier()->mutable_pipeline();
    }

    if (pipeline) {
        for (int i=0; i< pipeline->models_size(); i++) {
            downgradeSpecificationVersion(pipeline->mutable_models(i));
        }
    }

}

static bool isWeightParamOfType(const Specification::WeightParams &weight,
                                const WeightParamType& type) {
    return valueType(weight) == type;
}

static bool hasLSTMWeightParamOfType(const Specification::LSTMWeightParams& params,
                                     const WeightParamType& type) {

    return (isWeightParamOfType(params.inputgateweightmatrix(), type) ||
            isWeightParamOfType(params.forgetgateweightmatrix(), type) ||
            isWeightParamOfType(params.blockinputweightmatrix(), type) ||
            isWeightParamOfType(params.outputgateweightmatrix(), type) ||

            isWeightParamOfType(params.inputgaterecursionmatrix(), type) ||
            isWeightParamOfType(params.forgetgaterecursionmatrix(), type) ||
            isWeightParamOfType(params.blockinputrecursionmatrix(), type) ||
            isWeightParamOfType(params.outputgaterecursionmatrix(), type) ||

            isWeightParamOfType(params.inputgatebiasvector(), type) ||
            isWeightParamOfType(params.forgetgatebiasvector(), type) ||
            isWeightParamOfType(params.blockinputbiasvector(), type) ||
            isWeightParamOfType(params.outputgatebiasvector(), type) ||

            isWeightParamOfType(params.inputgatepeepholevector(), type) ||
            isWeightParamOfType(params.forgetgatepeepholevector(), type) ||
            isWeightParamOfType(params.outputgatepeepholevector(), type));
}

static bool hasWeightOfType(const Specification::NeuralNetworkLayer& layer,
                            const WeightParamType& type) {

    switch (layer.layer_case()) {
        case Specification::NeuralNetworkLayer::LayerCase::kConvolution:
            return (isWeightParamOfType(layer.convolution().weights(),type) ||
                    isWeightParamOfType(layer.convolution().bias(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kInnerProduct:
            return (isWeightParamOfType(layer.innerproduct().weights(),type) ||
                    isWeightParamOfType(layer.innerproduct().bias(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kBatchedMatmul:
            return (isWeightParamOfType(layer.batchedmatmul().weights(),type) ||
                    isWeightParamOfType(layer.batchedmatmul().bias(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kBatchnorm:
            return (isWeightParamOfType(layer.batchnorm().gamma(), type) ||
                    isWeightParamOfType(layer.batchnorm().beta(), type) ||
                    isWeightParamOfType(layer.batchnorm().mean(), type) ||
                    isWeightParamOfType(layer.batchnorm().variance(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kLoadConstant:
            return isWeightParamOfType(layer.loadconstant().data(), type);

        case Specification::NeuralNetworkLayer::LayerCase::kScale:
            return (isWeightParamOfType(layer.scale().scale(), type) ||
                    isWeightParamOfType(layer.scale().bias(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kSimpleRecurrent:
            return (isWeightParamOfType(layer.simplerecurrent().weightmatrix(), type) ||
                    isWeightParamOfType(layer.simplerecurrent().recursionmatrix(), type) ||
                    isWeightParamOfType(layer.simplerecurrent().biasvector(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kGru:
            return (isWeightParamOfType(layer.gru().updategateweightmatrix(), type) ||
                    isWeightParamOfType(layer.gru().resetgateweightmatrix(), type) ||
                    isWeightParamOfType(layer.gru().outputgateweightmatrix(), type) ||
                    isWeightParamOfType(layer.gru().updategaterecursionmatrix(), type) ||
                    isWeightParamOfType(layer.gru().resetgaterecursionmatrix(), type) ||
                    isWeightParamOfType(layer.gru().outputgaterecursionmatrix(), type) ||
                    isWeightParamOfType(layer.gru().updategatebiasvector(), type) ||
                    isWeightParamOfType(layer.gru().resetgatebiasvector(), type) ||
                    isWeightParamOfType(layer.gru().outputgatebiasvector(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kEmbedding:
            return (isWeightParamOfType(layer.embedding().weights(), type) ||
                    isWeightParamOfType(layer.embedding().bias(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kEmbeddingND:
            return (isWeightParamOfType(layer.embeddingnd().weights(), type) ||
                    isWeightParamOfType(layer.embeddingnd().bias(), type));

        case Specification::NeuralNetworkLayer::LayerCase::kUniDirectionalLSTM:
            return hasLSTMWeightParamOfType(layer.unidirectionallstm().weightparams(), type);

        case Specification::NeuralNetworkLayer::LayerCase::kBiDirectionalLSTM:
            return (hasLSTMWeightParamOfType(layer.bidirectionallstm().weightparams(0), type) ||
                    hasLSTMWeightParamOfType(layer.bidirectionallstm().weightparams(1), type));

        case Specification::NeuralNetworkLayer::LayerCase::kActivation:
            if(layer.activation().NonlinearityType_case() == Specification::ActivationParams::NonlinearityTypeCase::kPReLU) {
                return isWeightParamOfType(layer.activation().prelu().alpha(), type);
            } else if(layer.activation().NonlinearityType_case() == Specification::ActivationParams::NonlinearityTypeCase::kParametricSoftplus) {
                return (isWeightParamOfType(layer.activation().parametricsoftplus().alpha(), type) ||
                        isWeightParamOfType(layer.activation().parametricsoftplus().beta(), type));
            }
        default:
            break;
    }
    return false;
}

bool CoreML::hasFP16Weights(const Specification::Model& model) {
    // If any of the weight param is of type FP16, the model has FP16 weight
    return hasWeightOfType(model, FLOAT16);
}

static bool hasUnsignedQuantizedWeights(const Specification::Model& model) {
    return hasWeightOfType(model, QUINT);
}

static bool hasWeightOfType(const Specification::Model& model, const WeightParamType& wt) {
    return walkModelAndPipeline(
        model,
        [wt](const Specification::Model& m) {
            if (const auto *layers = getNNSpec(m)) {
                for (const auto& layer: *layers) {
                    if (hasWeightOfType(layer, wt)) {
                        return true;
                    }
                }
            }
            return false;
        });
}

// We'll check if the model has ONLY the IOS12 shape specifications
// if the old ones are also filled in with something plausible, then there is nothing
// preventing us from running on older versions of Core ML.
static bool hasFlexibleShapes(const Specification::Model& model) {
    return anyFeatureDescriptionOfModel(
        model,
        [](const auto& fd) {
            if (fd.type().Type_case() == Specification::FeatureType::kMultiArrayType &&
                fd.type().multiarraytype().ShapeFlexibility_case() != Specification::ArrayFeatureType::SHAPEFLEXIBILITY_NOT_SET) {
                return true;
            } else if (fd.type().Type_case() == Specification::FeatureType::kImageType &&
                       fd.type().imagetype().SizeFlexibility_case() != Specification::ImageFeatureType::SIZEFLEXIBILITY_NOT_SET) {
                return true;
            }
            return false;
        });
}

static bool hasFloat16MultiArray(const Specification::Model& model) {
    return anyFeatureDescriptionOfModel(
        model,
        [](const auto& fd) {
            return
                (fd.type().Type_case() == Specification::FeatureType::kMultiArrayType) &&
                (fd.type().multiarraytype().datatype() == Specification::ArrayFeatureType_ArrayDataType_FLOAT16);
        });
}

static bool hasOpsets(const Specification::Model& model, const std::string& opset) {
    return walkModelAndPipeline(
        model,
        [opset](const Specification::Model& m) {
            if (m.Type_case() != Specification::Model::kMlProgram) {
                return false;
            }

            auto main_iter = m.mlprogram().functions().find("main");
            if (main_iter != m.mlprogram().functions().end()) {
                const auto& main = main_iter->second;
                if (main.opset() == opset) {
                    return true;
                }
            }
            return false;
        });
}

static bool hasCoreML8Opsets(const Specification::Model& model) {
    return hasOpsets(model, "CoreML8");
}

static bool hasCoreML7Opsets(const Specification::Model& model) {
    return hasOpsets(model, "CoreML7");
}

static bool hasCoreML6Opsets(const Specification::Model& model) {
    return hasOpsets(model, "CoreML6");
}

static bool hasGrayscaleFloat16Image(const Specification::Model& model) {
    return anyFeatureDescriptionOfModel(
        model,
        [](const auto& fd) {
            return
                (fd.type().Type_case() == Specification::FeatureType::kImageType) &&
                (fd.type().imagetype().colorspace() == Specification::ImageFeatureType_ColorSpace_GRAYSCALE_FLOAT16);
        });
}

bool CoreML::hasIOS11_2Features(const Specification::Model& model) {
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            return
                hasCustomLayer(m) ||
                hasFP16Weights(m);
        });
}

bool CoreML::hasIOS12Features(const Specification::Model& model) {
    // New IOS12 features: flexible shapes, custom model, sequence feature type,
    // text classifier, word tagger, vision feature print, unsigned integer quantization
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            return
                hasFlexibleShapes(m) ||
                isCustomModel(m) ||
                hasCategoricalSequences(m) ||
                isAppleTextClassifier(m) ||
                isAppleWordTagger(m) ||
                isScenePrint(m) ||
                hasUnsignedQuantizedWeights(m) ||
                isNonMaxSuppression(m) ||
                isBayesianProbitRegressor(m) ||
                hasIOS12NewNeuralNetworkLayers(m);
        });
}

bool CoreML::hasIOS13Features(const Specification::Model& model) {
    // New IOS13 features:
    // - no constraint on rank for NN inputs
    // - model is marked as updatable
    //
    // - model is of type kKNearestNeighborsClassifier
    // - model is of type LinkedModel
    // - model is of type kItemSimilarityRecommender
    // - model is of sound analysis preprocessing
    //
    // - model is of type TextClassifier with revision == 2
    // - model is of type Gazetteer with revision == 2
    // - model is of type WordEmbedding with revision == 2

    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            if (m.isupdatable()) {
                return true;
            }

            switch (m.Type_case()) {
                case Specification::Model::kKNearestNeighborsClassifier:
                case Specification::Model::kLinkedModel:
                case Specification::Model::kItemSimilarityRecommender:
                case Specification::Model::kSoundAnalysisPreprocessing:
                    return true;
                case Specification::Model::kTextClassifier:
                    if (m.textclassifier().revision() == 2) {
                        return true;
                    }
                    break;
                case Specification::Model::kGazetteer:
                    if (m.gazetteer().revision() == 2) {
                        return true;
                    }
                    break;
                case Specification::Model::kWordEmbedding:
                    if (m.wordembedding().revision() == 2) {
                        return true;
                    }
                    break;
                default:
                    break;
            }
            return hasIOS13NeuralNetworkFeatures(m);
        });    
}

static bool hasDefaultValueForOptionalInputs(const Specification::Model& model) {
    // Checks if default optional value has been set or not
    return anyFeatureDescriptionOfModel(
        model,
        [](const auto& fd) {
            if (fd.type().Type_case() == Specification::FeatureType::kMultiArrayType && fd.type().isoptional()){
                switch (fd.type().multiarraytype().defaultOptionalValue_case()) {
                    case CoreML::Specification::ArrayFeatureType::kDoubleDefaultValue:
                    case CoreML::Specification::ArrayFeatureType::kFloatDefaultValue:
                    case CoreML::Specification::ArrayFeatureType::kIntDefaultValue:
                        return true;
                    default:
                        break;
                }
            }
            return false;
        });
}

static bool nonMaxSuppressionUsesFloat32InputsOrOutputs(const Specification::Model& model) {
    if (!isNonMaxSuppression(model)) {
        // not NMS.
        return false;
    }

    return anyFeatureDescriptionOfModel(
        model,
        [](const auto& fd) {
            return
                (fd.type().Type_case() == Specification::FeatureType::kMultiArrayType) &&
                (fd.type().multiarraytype().datatype() == Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
        });
}

bool CoreML::hasIOS14Features(const Specification::Model& model) {
    // New IOS14 features:
    // - new layers in Neural Network
    // - Non-zero values for optional inputs
    // - VisionFeaturePrint.Object
    // - Float32 input/output for Non-Maximum Suppression
    // - Apple Word Tagger using transfer learning (revision == 3)

    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            switch (m.Type_case()) {
                case Specification::Model::kSerializedModel:
                    // SerializedModel proto message was added in ios14
                    return true;
                case Specification::Model::kWordTagger:
                    if (m.wordtagger().revision() == 3) {
                        return true;
                    }
                    break;
                default:
                    break;
            }
            return
                hasIOS14NeuralNetworkFeatures(m) ||
                isObjectPrint(m) ||
                nonMaxSuppressionUsesFloat32InputsOrOutputs(m);
        });
}

bool CoreML::hasIOS15Features(const Specification::Model& model) {
    // New in IOS15 features:
    // - mlProgram proto message
    // - new sound print
    //
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            switch (m.Type_case()) {
                case Specification::Model::kMlProgram:
                    return true;
                case Specification::Model::kAudioFeaturePrint:
                    if (m.audiofeatureprint().has_sound()) {
                        return true;
                    }
                    break;
                default:
                    break;
            }
            return false;
        });
}

bool CoreML::hasIOS16Features(const Specification::Model& model) {
    // New in IOS16 features:
    //  - FLOAT16 array data type
    //  - GRAYSCALE_FLOAT16 image color space.
    //  - CoreML6 Opsets for mlProgram models
    //
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            return
                hasFloat16MultiArray(m) ||
                hasGrayscaleFloat16Image(m) ||
                hasCoreML6Opsets(m);
        });
}

bool CoreML::hasIOS17Features(const Specification::Model& model) {
    // New in IOS17 features:
    // - Revision 2 of Apple Vision feature extractor for scenes
    // - BERT embedding for text classifier and word tagger (revision == 4)

    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            switch (m.Type_case()) {
                case Specification::Model::kVisionFeaturePrint:
                    if (m.visionfeatureprint().has_scene() &&
                        m.visionfeatureprint().scene().version() == 2) {
                        return true;
                    }
                    break;
                case Specification::Model::kClassConfidenceThresholding:
                    return true;
                case Specification::Model::kWordTagger:
                    if (m.wordtagger().revision() == 4) {
                        return true;
                    }
                    break;
                case Specification::Model::kTextClassifier:
                    if (m.textclassifier().revision() == 4) {
                        return true;
                    }
                    break;
                case Specification::Model::kMlProgram:
                    if (hasCoreML7Opsets(m)) {
                        return true;
                    }
                    break;
                default:
                    break;
            }
            return false;
        });
}

bool CoreML::hasIOS18Features(const Specification::Model& model) {
    // New in IOS18 features:
    // - Language expansion for multilingual BERT used in text classifier and word tagger (revision == 5)

    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            if (hasMultiFunctions(m) ||
                hasEmptyInput(m)) {
                return true;
            }

            switch (m.Type_case()) {
                case Specification::Model::kWordTagger:
                    if (m.wordtagger().revision() == 5) {
                        return true;
                    }
                    break;
                case Specification::Model::kTextClassifier:
                    if (m.textclassifier().revision() == 5) {
                        return true;
                    }
                    break;
                case Specification::Model::kMlProgram:
                    if (hasCoreML8Opsets(m)) {
                        return true;
                    }
                    break;
                default:
                    break;
            }
            return false;
        });
}

static bool isCustomModel(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kCustomModel);
}

static bool isAppleWordTagger(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kWordTagger);
}

static bool isAppleTextClassifier(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kTextClassifier);
}

static bool isAppleGazetteer(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kGazetteer);
}

static bool isAppleWordEmbedding(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kWordEmbedding);
}

bool CoreML::hasAppleImageFeatureExtractor(const Specification::Model& model) {
    return walkModelAndPipeline(
        model,
        [](const auto& m) {
            return m.Type_case() == Specification::Model::kVisionFeaturePrint;
        });
}

static bool isScenePrint(const Specification::Model& model) {
    return
        model.Type_case() == Specification::Model::kVisionFeaturePrint &&
        model.visionfeatureprint().has_scene();
}

static bool isObjectPrint(const Specification::Model& model) {
    return
        model.Type_case() == Specification::Model::kVisionFeaturePrint &&
        model.visionfeatureprint().has_objects();
}

static bool isAppleAudioFeatureExtractor(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kAudioFeaturePrint);
}

static bool isSoundPrint(const Specification::Model& model) {
    return (isAppleAudioFeatureExtractor(model) && model.audiofeatureprint().has_sound());
}

static bool isNonMaxSuppression(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kNonMaximumSuppression);
}

static bool isBayesianProbitRegressor(const Specification::Model& model) {
    return (model.Type_case() == Specification::Model::kBayesianProbitRegressor);
}

static bool hasCategoricalSequences(const Specification::Model& model) {
    return anyFeatureDescriptionOfModel(
        model,
        [](const auto& fd) {
            if (fd.type().Type_case() == Specification::FeatureType::kSequenceType) {
                switch (fd.type().sequencetype().Type_case()) {
                    case Specification::SequenceFeatureType::kStringType:
                    case Specification::SequenceFeatureType::kInt64Type:
                        return true;
                    default:
                        break;
                }
            }
            return false;
        });
}

static bool hasIOS12NewNeuralNetworkLayers(const Specification::Model& model) {

    // Return True if the model has the two new NN layers added in iOS 12, which are
    // resizeBilinear and CropResize

    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            if (const auto *layers = getNNSpec(m)) {
                for (const auto& layer: *layers) {
                    if (layer.layer_case() == Specification::NeuralNetworkLayer::kResizeBilinear ||
                        layer.layer_case() == Specification::NeuralNetworkLayer::kCropResize) {
                        return true;
                    }
                }
            }
            return false;
        });
}

bool CoreML::isIOS12NeuralNetworkLayer(const Specification::NeuralNetworkLayer& layer) {

    // Return True if the NN layer is from the set exposed in iOS 12
    switch (layer.layer_case()) {
        case Specification::NeuralNetworkLayer::LayerCase::kConvolution:
            return (layer.input().size() == 1);
        case Specification::NeuralNetworkLayer::LayerCase::kInnerProduct:
            return !layer.innerproduct().int8dynamicquantize();
        case Specification::NeuralNetworkLayer::LayerCase::kBatchnorm:
        case Specification::NeuralNetworkLayer::LayerCase::kActivation:
        case Specification::NeuralNetworkLayer::LayerCase::kPooling:
        case Specification::NeuralNetworkLayer::LayerCase::kPadding:
        case Specification::NeuralNetworkLayer::LayerCase::kConcat:
        case Specification::NeuralNetworkLayer::LayerCase::kLrn:
        case Specification::NeuralNetworkLayer::LayerCase::kSoftmax:
        case Specification::NeuralNetworkLayer::LayerCase::kSplit:
        case Specification::NeuralNetworkLayer::LayerCase::kAdd:
        case Specification::NeuralNetworkLayer::LayerCase::kMultiply:
        case Specification::NeuralNetworkLayer::LayerCase::kUnary:
        case Specification::NeuralNetworkLayer::LayerCase::kUpsample:
            if (layer.upsample().linearupsamplemode() != Specification::UpsampleLayerParams_LinearUpsampleMode_DEFAULT) {
                return false;
            }
            if (layer.upsample().fractionalscalingfactor_size() > 0) {
                return false;
            }
        case Specification::NeuralNetworkLayer::LayerCase::kBias:
        case Specification::NeuralNetworkLayer::LayerCase::kL2Normalize:
        case Specification::NeuralNetworkLayer::LayerCase::kReshape:
        case Specification::NeuralNetworkLayer::LayerCase::kFlatten:
        case Specification::NeuralNetworkLayer::LayerCase::kPermute:
        case Specification::NeuralNetworkLayer::LayerCase::kReduce:
        case Specification::NeuralNetworkLayer::LayerCase::kLoadConstant:
        case Specification::NeuralNetworkLayer::LayerCase::kScale:
        case Specification::NeuralNetworkLayer::LayerCase::kSimpleRecurrent:
        case Specification::NeuralNetworkLayer::LayerCase::kGru:
        case Specification::NeuralNetworkLayer::LayerCase::kUniDirectionalLSTM:
        case Specification::NeuralNetworkLayer::LayerCase::kBiDirectionalLSTM:
        case Specification::NeuralNetworkLayer::LayerCase::kCrop:
        case Specification::NeuralNetworkLayer::LayerCase::kAverage:
        case Specification::NeuralNetworkLayer::LayerCase::kMax:
        case Specification::NeuralNetworkLayer::LayerCase::kMin:
        case Specification::NeuralNetworkLayer::LayerCase::kDot:
        case Specification::NeuralNetworkLayer::LayerCase::kMvn:
        case Specification::NeuralNetworkLayer::LayerCase::kEmbedding:
        case Specification::NeuralNetworkLayer::LayerCase::kSequenceRepeat:
        case Specification::NeuralNetworkLayer::LayerCase::kReorganizeData:
            if (layer.reorganizedata().mode() == Specification::ReorganizeDataLayerParams::PIXEL_SHUFFLE) {
                      return false;
            }
        case Specification::NeuralNetworkLayer::LayerCase::kSlice:
        case Specification::NeuralNetworkLayer::LayerCase::kCustom:
        case Specification::NeuralNetworkLayer::kResizeBilinear:
        case Specification::NeuralNetworkLayer::kCropResize:
            return true;
        default:
            return false;
    }
}

static bool hasIOS13NeuralNetworkFeatures(const Specification::Model& model) {

    /* check if any of the messages in NeuralNetwork.proto, that were added in iOS version 13, are being used.
      If they are, return True, otherwise return False.

     In particular, check for the presence of the following messages:
     1. any new layer type, which was not in iOS 12.
     2. if the value of enums "NeuralNetworkMultiArrayShapeMapping" or "NeuralNetworkImageShapeMapping" is non 0
     */
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            if (m.Type_case() == Specification::Model::TypeCase::kNeuralNetwork) {
                if (m.neuralnetwork().arrayinputshapemapping() != Specification::NeuralNetworkMultiArrayShapeMapping::RANK5_ARRAY_MAPPING) {
                    return true;
                }
                if (m.neuralnetwork().imageinputshapemapping() != Specification::NeuralNetworkImageShapeMapping::RANK5_IMAGE_MAPPING) {
                    return true;
                }
            } else if (m.Type_case() == Specification::Model::TypeCase::kNeuralNetworkRegressor) {
                if (m.neuralnetworkregressor().arrayinputshapemapping() != Specification::NeuralNetworkMultiArrayShapeMapping::RANK5_ARRAY_MAPPING) {
                    return true;
                }
                if (m.neuralnetworkregressor().imageinputshapemapping() != Specification::NeuralNetworkImageShapeMapping::RANK5_IMAGE_MAPPING) {
                    return true;
                }
            } else if (m.Type_case() == Specification::Model::TypeCase::kNeuralNetworkClassifier) {
                if (m.neuralnetworkclassifier().arrayinputshapemapping() != Specification::NeuralNetworkMultiArrayShapeMapping::RANK5_ARRAY_MAPPING) {
                    return true;
                }
                if (m.neuralnetworkclassifier().imageinputshapemapping() != Specification::NeuralNetworkImageShapeMapping::RANK5_IMAGE_MAPPING) {
                    return true;
                }
            }

            // check for new layers: by checking if its NOT one of the layers supported in iOS 12
            if (const auto *layers = getNNSpec(m)) {
                for (const auto& layer: *layers) {
                    if (!isIOS12NeuralNetworkLayer(layer)) {
                        return true;
                    }
                }
            }
            return false;
        });
}

static bool hasIOS14NeuralNetworkFeatures(const Specification::Model& model) {

    // Return True if the model has the new Neural network features added in
    // ios 14
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            if (hasDefaultValueForOptionalInputs(m)) {
                return true;
            }

            const auto *layers = getNNSpec(m);
            if (!layers) {
                return false;
            }

            for (const auto& layer: *layers) {
                switch (layer.layer_case()) {
                    case Specification::NeuralNetworkLayer::kCumSum:
                    case Specification::NeuralNetworkLayer::kOneHot:
                    case Specification::NeuralNetworkLayer::kClampedReLU:
                    case Specification::NeuralNetworkLayer::kArgSort:
                    case Specification::NeuralNetworkLayer::kPooling3D:
                    case Specification::NeuralNetworkLayer::kGlobalPooling3D:
                    case Specification::NeuralNetworkLayer::kSliceBySize:
                    case Specification::NeuralNetworkLayer::kConvolution3D:
                        return true;
                    case Specification::NeuralNetworkLayer::kSliceDynamic:
                        if (layer.input().size() == 7) {
                            return true;
                        } else if (layer.slicedynamic().squeezemasks_size()) {
                            return true;
                        }
                    case Specification::NeuralNetworkLayer::kUpsample:
                        if (layer.upsample().linearupsamplemode() != Specification::UpsampleLayerParams_LinearUpsampleMode_DEFAULT) {
                            return true;
                        }
                        if (layer.upsample().fractionalscalingfactor_size() > 0) {
                            return true;
                        }
                    case Specification::NeuralNetworkLayer::kReorganizeData:
                        if (layer.reorganizedata().mode() == Specification::ReorganizeDataLayerParams::PIXEL_SHUFFLE) {
                            return true;
                        }
                    case Specification::NeuralNetworkLayer::kInnerProduct:
                        if (layer.innerproduct().int8dynamicquantize()) {
                            return true;
                        }
                    case Specification::NeuralNetworkLayer::kBatchedMatmul:
                        if (layer.batchedmatmul().int8dynamicquantize()) {
                            return true;
                        }
                    case Specification::NeuralNetworkLayer::kConcatND:
                        if (layer.concatnd().interleave()) {
                            return true;
                        }
                    default:
                        continue;
                }
            }
            return false;
        });
}

static bool hasMultiFunctions(const Specification::Model& model) {
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            const auto& description = m.description();
            return description.functions_size() != 0 || !description.defaultfunctionname().empty();
        });
}

static bool hasEmptyInput(const Specification::Model& model) {
    return walkModelAndPipeline(
        model,
        [](const Specification::Model& m) {
            const auto& modelDescription = m.description();
            if (modelDescription.functions_size() == 0) {
                // A single function syntax
                if (modelDescription.input_size() == 0) {
                    return true;
                }
            } else {
                // A multi function syntax
                for (const auto& functionDescription: modelDescription.functions()) {
                    if (functionDescription.input_size() == 0) {
                        return true;
                    }
                }
            }
            return false;
        });
}
