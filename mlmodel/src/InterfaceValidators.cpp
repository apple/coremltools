//
//  InterfaceValidators.cpp
//  libmlmodelspec
//
//  Created by Michael Siracusa on 12/8/16.
//  Copyright © 2016 Apple. All rights reserved.
//

#include "Result.hpp"
#include "Validators.hpp"
#include "ValidatorUtils-inl.hpp"
#include "../build/format/Model.pb.h"

namespace CoreML {

    Result validateFeatureDescription(const Specification::FeatureDescription& desc, bool isInput) {
        if (desc.name() == "") {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Feature description must have a non-empty name.");
        }

        if (!desc.has_type()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Feature description " + desc.name() + " must specify a valid feature type.");
        }

        const auto& type = desc.type();
        switch (type.Type_case()) {
            case Specification::FeatureType::kDoubleType:
            case Specification::FeatureType::kInt64Type:
            case Specification::FeatureType::kStringType:
                // in general, non-parametric types need no further validation
                break;

            case Specification::FeatureType::kMultiArrayType:
            {
                if (isInput && type.multiarraytype().shape_size() == 0) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of multiarray feature '" + desc.name() + "' has an invalid shape. "
                                  "The shape has zero-length or is missing.");
                }

                for (int i=0; i < type.multiarraytype().shape_size(); i++) {
                    const auto &value = type.multiarraytype().shape(i);
                    if (value < 0) {
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of multiarray feature '" + desc.name() + "' has an invalid shape. "
                                      "Element " + std::to_string(i) + " has non-positive value " + std::to_string(value) + ".");
                    }
                }

                switch (type.multiarraytype().datatype()) {
                    case Specification::ArrayFeatureType_ArrayDataType_DOUBLE:
                    case Specification::ArrayFeatureType_ArrayDataType_FLOAT32:
                    case Specification::ArrayFeatureType_ArrayDataType_INT32:
                        break;
                    default:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of multiarray feature '" + desc.name() + "' has an invalid or unspecified dataType. "
                                      "It must be specified as DOUBLE, FLOAT32 or INT32");
                }
                break;
            }
            case Specification::FeatureType::kDictionaryType:
                switch (type.dictionarytype().KeyType_case()) {
                    case Specification::DictionaryFeatureType::KeyTypeCase::kInt64KeyType:
                    case Specification::DictionaryFeatureType::KeyTypeCase::kStringKeyType:
                        break;
                    case Specification::DictionaryFeatureType::KeyTypeCase::KEYTYPE_NOT_SET:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of dictionary feature '" + desc.name() + "' must contain a key type of either Int64 or String.");
                }
                break;

            case Specification::FeatureType::kImageType:

                if (type.imagetype().width() <= 0) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of image feature '" + desc.name() +
                                  "' has missing or non-positive width " + std::to_string(type.imagetype().width()) + ".");
                }

                if (type.imagetype().height() <= 0) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of image feature '" + desc.name() +
                                  "' has missing or non-positive height " + std::to_string(type.imagetype().width()) + ".");
                }

                switch (type.imagetype().colorspace()) {
                    case Specification::ImageFeatureType_ColorSpace_GRAYSCALE:
                    case Specification::ImageFeatureType_ColorSpace_RGB:
                    case Specification::ImageFeatureType_ColorSpace_BGR:
                        break;
                    default:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of image feature '" + desc.name() +
                                      "' has missing or invalid colorspace. It must be RGB, BGR or GRAYSCALE.");
                }
                break;

            case Specification::FeatureType::TYPE_NOT_SET:
                // these aren't equal to anything, even themselves
                return Result(ResultType::INVALID_MODEL_INTERFACE,
                              "Feature description has an unspecified or invalid type for feature '" + desc.name() + "'.");
        }

        // If we got here, the feature description is valid.
        return Result();
    }

    Result validateFeatureDescriptions(const Specification::ModelDescription& interface) {
        // a model must have at least one input and one output
        if (interface.input_size() < 1) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Models must have one or more inputs.");
        }
        if (interface.output_size() < 1) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Models must have one or more outputs.");
        }

        for (const auto& input : interface.input()) {
            Result r = validateFeatureDescription(input,true);
            if (!r.good()) { return r; }
        }

        for (const auto& output : interface.output()) {
            Result r = validateFeatureDescription(output,false);
            if (!r.good()) { return r; }
        }

        // If we got here, all inputs/outputs seem good independently of each other.
        return Result();
    }

    Result validateModelDescription(const Specification::ModelDescription& interface) {

        Result result = validateFeatureDescriptions(interface);
        if (!result.good()) {
            return result;
        }

        return result;

    }


    Result validateRegressorInterface(const Specification::ModelDescription& description) {
        
        if (description.predictedfeaturename() == "") {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Specification is missing regressor predictedFeatureName.");
        }
        
        // Validate feature descriptions
        Result result = validateFeatureDescriptions(description);
        if (!result.good()) {
            return result;
        }

        result = validateDescriptionsContainFeatureWithNameAndType(description.output(),
                                                                   description.predictedfeaturename(),
                                                                   {Specification::FeatureType::kDoubleType, Specification::FeatureType::kMultiArrayType});
        if (!result.good()) {
            return result;
        }
        return result;
    }

    /*
     * Validate optional inputs/outputs.
     * For most models, optional is not allowed (all inputs/outputs required).
     * Some will template-specialize to override.
     */
    inline Result validateOptionalOutputs(const Specification::ModelDescription& interface) {
        for (const auto& output : interface.output()) {
            if (output.type().isoptional()) {
                return Result(ResultType::INVALID_MODEL_INTERFACE, "Outputs cannot be optional.");
            }
        }
        return Result();
    }

    static Result validateOptionalGeneric(const Specification::ModelDescription& interface) {
        for (const auto& input : interface.input()) {
            if (input.type().isoptional()) {
                return Result(ResultType::INVALID_MODEL_PARAMETERS, "Features cannot be optional to this type of model.");
            }
        }
        return validateOptionalOutputs(interface);
    }
    
    inline Result validateOptionalTree(const Specification::ModelDescription& interface) {
        return validateOptionalOutputs(interface);
    }
    
    inline Result validateOptionalNN(const Specification::ModelDescription& description) {
        // just need to check that not all inputs are optional
        bool hasNotOptional = false;
        for (const auto& input : description.input()) {
            if (!input.type().isoptional()) {
                hasNotOptional = true;
                break;
            }
        }
        if (!hasNotOptional) {
            return Result(ResultType::INVALID_MODEL_PARAMETERS, "At least one feature for a neural network must NOT be optional.");
        }
        return Result();
    }

    Result validateOptional(const Specification::Model& format) {
        Result r;
        switch (format.Type_case()) {
            case Specification::Model::kImputer:
                // Imputed values can be handled by replacing a particular value, so
                // optional is not required.
                break;
            case Specification::Model::kNeuralNetwork:
            case Specification::Model::kNeuralNetworkRegressor:
            case Specification::Model::kNeuralNetworkClassifier:
                r = validateOptionalNN(format.description());
                break;
            case Specification::Model::kTreeEnsembleRegressor:
            case Specification::Model::kTreeEnsembleClassifier:
                // allow arbitrary optional in tree inputs, just check outputs
                break;
            case Specification::Model::kPipeline:
            case Specification::Model::kPipelineRegressor:
            case Specification::Model::kPipelineClassifier:
                // pipeline has valid optional inputs iff the models inside are valid.
                // this should be guaranteed by the pipeline validator.
                break;
            case Specification::Model::kIdentity:
                // anything goes for the identity function
                break;
            default:
                r = validateOptionalGeneric(format.description());
        }
        if (!r.good()) {
            return r;
        }

        return validateOptionalOutputs(format.description());
    }

}
