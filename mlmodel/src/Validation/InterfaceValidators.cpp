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
#include "Globals.hpp"
#include "Model.hpp"
#include "Utils.hpp"

namespace CoreML {

    ValidationPolicy::ValidationPolicy(MLModelType modelType)
        :allowsEmptyInput(Model::modelTypeAllowsEmptyInput(modelType)),
         allowsEmptyOutput(false),
         allowsMultipleFunctions(Model::modelTypeAllowsMultipleFunctions(modelType)),
         allowsStatefulPrediction(Model::modelTypeAllowsStatefulPrediction(modelType))
    {}

    Result validateSizeRange(const Specification::SizeRange &range) {
        if (range.upperbound() > 0 && range.lowerbound() > static_cast<unsigned long long>(range.upperbound())) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Size range is invalid (" + std::to_string(range.lowerbound()) + ", " + std::to_string(range.upperbound()) + " ).");
        }
        return Result();
    }

    Result validateFeatureDescription(const Specification::FeatureDescription& desc, int modelVersion, FeatureIOType featureIOType) {
        if (desc.name() == "") {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Feature description must have a non-empty name.");
        }

        if (!desc.has_type()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Feature description " + desc.name() + " must specify a valid feature type.");
        }

        const auto& type = desc.type();
        if (type.Type_case() == Specification::FeatureType::kStateType) {
            // State features must be declared in the state feature descriptions. (For backward compatibility reason,
            // it's also allowed to be in the input feature description.)
            if (featureIOType != FeatureIOType::STATE && featureIOType != FeatureIOType::INPUT) {
                return Result(ResultType::INVALID_MODEL_INTERFACE,
                              "State feature '" + desc.name() + "' should only be declared in the state feature description.");
            }
        } else {
            // State feature description shall not have anything but state features.
            if (featureIOType == FeatureIOType::STATE) {
                return Result(ResultType::INVALID_MODEL_INTERFACE,
                              "State feature description can declare only state features, but '" + desc.name() + "' is not.");
            }
        }

        switch (type.Type_case()) {
            case Specification::FeatureType::kDoubleType:
            case Specification::FeatureType::kInt64Type:
            case Specification::FeatureType::kStringType:
                // in general, non-parametric types need no further validation
                break;

            case Specification::FeatureType::kMultiArrayType:
            {
                const auto &defaultShape = type.multiarraytype().shape();
                bool hasExplicitDefault = (type.multiarraytype().shape_size() != 0);
                bool hasImplictDefault = false;
                // Newer versions use the updated shape constraints for images and multi-arrays
                if (modelVersion >= MLMODEL_SPECIFICATION_VERSION_IOS12) {

                    switch (type.multiarraytype().ShapeFlexibility_case()) {

                        case Specification::ArrayFeatureType::kEnumeratedShapes: {

                            hasImplictDefault = true;

                            if (type.multiarraytype().enumeratedshapes().shapes_size() == 0) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of multiarray feature '" + desc.name() + "' has enumerated zero permitted sizes.");
                            }

                            for (auto &shape : type.multiarraytype().enumeratedshapes().shapes()) {
                                if (shape.shape_size() == 0) {
                                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                                  "Description of multiarray feature '" + desc.name() + "' has enumerated shapes with zero dimensions.");
                                }
                            }

                            if (!hasExplicitDefault) {
                                break;
                            }

                            bool foundDefault = false;
                            for (auto &shape : type.multiarraytype().enumeratedshapes().shapes()) {
                                if (shape.shape_size() != defaultShape.size()) { continue; }
                                foundDefault = true;
                                for (int d = 0; d < shape.shape_size(); d++) {
                                    if (defaultShape[d] != shape.shape(d)) { foundDefault = false; break; }
                                }
                                if (foundDefault) { break; }
                            }

                            if (!foundDefault) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of multiarray feature '" + desc.name() + "' has a default shape specified " +
                                              " which is not within the allowed enumerated shapes specified.");
                            }

                            break;
                        }
                        case Specification::ArrayFeatureType::kShapeRange: {

                            hasImplictDefault = true;

                            const auto& sizeRanges = type.multiarraytype().shaperange().sizeranges();
                            for (int i = 0; i < sizeRanges.size(); i++) {
                                const auto &range = sizeRanges[i];
                                Result res = validateSizeRange(range);
                                if (!res.good()) {
                                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                                  "Description of multiarray feature '" + desc.name() +
                                                  "' has an invalid range for dimension " + std::to_string(i) + ". " +
                                                  res.message());
                                }
                            }

                            if (!hasExplicitDefault) {
                                break;
                            }

                            // Check if default is compatible
                            if (defaultShape.size() != sizeRanges.size()) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of multiarray feature '" + desc.name() +
                                              "' has a default " + std::to_string(defaultShape.size()) + "-d shape but a " +
                                              std::to_string(sizeRanges.size()) + "-d shape range");
                            }

                             for (int i = 0; i < sizeRanges.size(); i++) {
                                 if (defaultShape[i] < (int)sizeRanges[i].lowerbound() ||
                                     (sizeRanges[i].upperbound() >= 0 && defaultShape[i] > sizeRanges[i].upperbound())) {

                                     return Result(ResultType::INVALID_MODEL_INTERFACE,
                                                   "Description of multiarray feature '" + desc.name() +
                                                   "' has a default shape that is out of the specified shape range");
                                 }
                             }

                            break;
                        }
                        case Specification::ArrayFeatureType::SHAPEFLEXIBILITY_NOT_SET:
                            break;
                    }

                }

                if (featureIOType == FeatureIOType::INPUT && !hasExplicitDefault && !hasImplictDefault) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of multiarray feature '" + desc.name() + "' has missing shape constraints.");
                }

                if (hasExplicitDefault) {
                    for (int i=0; i < type.multiarraytype().shape_size(); i++) {
                        const auto &value = type.multiarraytype().shape(i);
                        if (value < 0) {
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Description of multiarray feature '" + desc.name() + "' has an invalid shape. "
                                          "Element " + std::to_string(i) + " has non-positive value " + std::to_string(value) + ".");
                        }
                    }
                }

                switch (type.multiarraytype().datatype()) {
                    case Specification::ArrayFeatureType_ArrayDataType_DOUBLE:
                    case Specification::ArrayFeatureType_ArrayDataType_FLOAT32:
                    case Specification::ArrayFeatureType_ArrayDataType_INT32:
                        break;
                    case Specification::ArrayFeatureType_ArrayDataType_FLOAT16:
                        if (modelVersion < MLMODEL_SPECIFICATION_VERSION_IOS16) {
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Description of multiarray feature '" + desc.name() +
                                          "' has FLOAT16 dataType, which is only valid in specification version >= " += std::to_string(MLMODEL_SPECIFICATION_VERSION_IOS16)+
                                          ". This model has version " + std::to_string(modelVersion));
                        }

                        break;
                    default:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of multiarray feature '" + desc.name() + "' has an invalid or unspecified dataType. "
                                      "It must be specified as DOUBLE, FLOAT32, FLOAT16 or INT32");
                }

                switch (type.multiarraytype().defaultOptionalValue_case()) {
                    case CoreML::Specification::ArrayFeatureType::kDoubleDefaultValue:
                        if (type.multiarraytype().datatype() != Specification::ArrayFeatureType_ArrayDataType_DOUBLE){
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Description of multiarray feature '" + desc.name() + "' has mistmatch"
                                          " between dataType and the type of default optional value.");
                        }
                        break;
                    case CoreML::Specification::ArrayFeatureType::kFloatDefaultValue:
                        if (type.multiarraytype().datatype() != Specification::ArrayFeatureType_ArrayDataType_FLOAT32 &&
                            type.multiarraytype().datatype() != Specification::ArrayFeatureType_ArrayDataType_FLOAT16){
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Description of multiarray feature '" + desc.name() + "' has mistmatch"
                                          " between dataType and the type of default optional value.");
                        }
                        break;
                    case CoreML::Specification::ArrayFeatureType::kIntDefaultValue:
                        if (type.multiarraytype().datatype() != Specification::ArrayFeatureType_ArrayDataType_INT32){
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Description of multiarray feature '" + desc.name() + "' has mistmatch"
                                          " between dataType and the type of default optional value.");
                        }
                        break;
                    default:
                        break;
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

            case Specification::FeatureType::kImageType: {

                int64_t defaultWidth = type.imagetype().width();
                int64_t defaultHeight = type.imagetype().height();
                bool hasDefault = (defaultWidth > 0 && defaultHeight > 0);

                if (modelVersion >= MLMODEL_SPECIFICATION_VERSION_IOS12) {

                    switch (type.imagetype().SizeFlexibility_case()) {

                        case Specification::ImageFeatureType::kEnumeratedSizes: {

                            if (type.imagetype().enumeratedsizes().sizes_size() == 0) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of image feature '" + desc.name() + "' has enumerated zero permitted sizes.");
                            }

                            if (!hasDefault) {
                                defaultWidth = (int64_t)type.imagetype().enumeratedsizes().sizes(0).width();
                                defaultHeight = (int64_t)type.imagetype().enumeratedsizes().sizes(0).height();
                                break;
                            }

                            bool foundDefault = false;
                            for (auto &size : type.imagetype().enumeratedsizes().sizes()) {
                                if (defaultWidth == (int64_t)size.width() && defaultHeight == (int64_t)size.height()) {
                                    foundDefault = true;
                                    break;
                                }
                            }

                            if (!foundDefault) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of image feature '" + desc.name() + "' has a default size of " +
                                              std::to_string(defaultWidth) + " × " + std::to_string(defaultHeight) +
                                              " which is not within the allowed enumerated sizes specified.");
                            }

                            break;
                        }
                        case Specification::ImageFeatureType::kImageSizeRange:
                        {
                            const auto& widthRange = type.imagetype().imagesizerange().widthrange();
                            Result res = validateSizeRange(widthRange);
                            if (!res.good()) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of image feature '" + desc.name() + "' has an invalid flexible width range. "
                                               + res.message());
                            }

                            const auto& heightRange = type.imagetype().imagesizerange().heightrange();
                            res = validateSizeRange(heightRange);
                            if (!res.good()) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of image feature '" + desc.name() + "' has an invalid flexible height range. "
                                              + res.message());
                            }

                            if (!hasDefault) {
                                defaultWidth = (int64_t)widthRange.lowerbound();
                                defaultHeight = (int64_t)heightRange.lowerbound();
                                break;
                            }

                            if (defaultWidth < (int64_t)widthRange.lowerbound() ||
                                (widthRange.upperbound() >=0 && defaultWidth > widthRange.upperbound())) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of image feature '" + desc.name() + "' default width "
                                              + std::to_string(defaultWidth) + " is not within specified flexible width range");
                            }

                            if (defaultHeight < (int64_t)heightRange.lowerbound() ||
                                (heightRange.upperbound() >=0 && defaultHeight > heightRange.upperbound())) {
                                return Result(ResultType::INVALID_MODEL_INTERFACE,
                                              "Description of image feature '" + desc.name() + "' default height "
                                              + std::to_string(defaultHeight) + " is not within specified flexible height range");
                            }


                            break;
                        }
                        default:
                            break;
                    }

                }

                if (defaultWidth <= 0) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of image feature '" + desc.name() +
                                  "' has missing or non-positive width " + std::to_string(type.imagetype().width()) + ".");
                }

                if (defaultHeight <= 0) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of image feature '" + desc.name() +
                                  "' has missing or non-positive height " + std::to_string(type.imagetype().height()) + ".");
                }

                switch (type.imagetype().colorspace()) {
                    case Specification::ImageFeatureType_ColorSpace_GRAYSCALE:
                    case Specification::ImageFeatureType_ColorSpace_RGB:
                    case Specification::ImageFeatureType_ColorSpace_BGR:
                        break;
                    case Specification::ImageFeatureType_ColorSpace_GRAYSCALE_FLOAT16:
                        if (modelVersion < MLMODEL_SPECIFICATION_VERSION_IOS16) {
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Description of image feature '" + desc.name() +
                                          "' has GRAYSCALE_FLOAT16 colorspace, which is only valid in specification version >= " += std::to_string(MLMODEL_SPECIFICATION_VERSION_IOS16)+
                                          ". This model has version " + std::to_string(modelVersion));
                        }
                        break;
                    default:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of image feature '" + desc.name() +
                                      "' has missing or invalid colorspace. It must be RGB, BGR or GRAYSCALE.");
                }
                break;
            }
            case Specification::FeatureType::kSequenceType: {

                if (modelVersion < MLMODEL_SPECIFICATION_VERSION_IOS12) {
                    return  Result(ResultType::INVALID_MODEL_INTERFACE,
                                   "Sequence types are only valid in specification verison >= " + std::to_string(MLMODEL_SPECIFICATION_VERSION_IOS12)+
                                   ". This model has version " + std::to_string(modelVersion));
                }

                // Validate size
                Result res = validateSizeRange(type.sequencetype().sizerange());
                if (!res.good()) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of sequence feature '" + desc.name() + "' has invalid allowed sizes. "
                                  + res.message());
                }

                // Validate type

                switch (type.sequencetype().Type_case()) {
                    case Specification::SequenceFeatureType::kInt64Type:
                    case Specification::SequenceFeatureType::kStringType:
                        break;
                    case Specification::SequenceFeatureType::TYPE_NOT_SET:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of sequence feature '" + desc.name() + "' has invalid or missing type. "
                                      "Only Int64 and String sequences are currently supported");


                }
                break;
            }
            case Specification::FeatureType::kStateType:
            {
                if (modelVersion < MLMODEL_SPECIFICATION_VERSION_IOS18) {
                    return  Result(ResultType::INVALID_MODEL_INTERFACE,
                                   "State types are only valid in specification verison >= " + std::to_string(MLMODEL_SPECIFICATION_VERSION_IOS18)+
                                   ". This model has version " + std::to_string(modelVersion));
                }

                if (type.isoptional()) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "State feature '" + desc.name() + "' cannot be optional.");
                }

                const auto &defaultShape = type.statetype().arraytype().shape();
                bool hasExplicitDefault = (type.statetype().arraytype().shape_size() != 0);

                if (!hasExplicitDefault) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "Description of State feature '" + desc.name() + "' has missing shape constraints.");
                }

                for (int i=0; i < type.statetype().arraytype().shape_size(); i++) {
                    const auto &value = type.statetype().arraytype().shape(i);
                    if (value < 0) {
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of State feature '" + desc.name() + "' has an invalid shape. "
                                      "Element " + std::to_string(i) + " has non-positive value " + std::to_string(value) + ".");
                    }
                }

                switch (type.statetype().arraytype().datatype()) {
                    case Specification::ArrayFeatureType_ArrayDataType_FLOAT16:
                        break;
                    default:
                        return Result(ResultType::INVALID_MODEL_INTERFACE,
                                      "Description of State feature '" + desc.name() + "' has an invalid or unspecified dataType. "
                                      "It must be specified as FLOAT16");
                }

                if (type.statetype().arraytype().ShapeFlexibility_case() != Specification::ArrayFeatureType::SHAPEFLEXIBILITY_NOT_SET) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "State feature '" + desc.name() + "' cannot have flexible shape.");
                }

                if (type.statetype().arraytype().defaultOptionalValue_case() != Specification::ArrayFeatureType::DEFAULTOPTIONALVALUE_NOT_SET) {
                    return Result(ResultType::INVALID_MODEL_INTERFACE,
                                  "State feature '" + desc.name() + "' cannot have default optional value.");
                }
                break;
            }
            case Specification::FeatureType::TYPE_NOT_SET:
                // these aren't equal to anything, even themselves
                return Result(ResultType::INVALID_MODEL_INTERFACE,
                              "Feature description has an unspecified or invalid type for feature '" + desc.name() + "'.");
        }

        // If we got here, the feature description is valid.
        return Result();
    }

    inline Result validateModelLevelFeatureDescriptionsAreEmpty(const Specification::ModelDescription& interface) {
        if (interface.input_size() != 0) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Multi-function model must not use top level input feature description.");
        }

        if (interface.output_size() != 0) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Multi-function model must not use top level output feature description.");
        }

        if (interface.state_size() != 0) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Multi-function model must not use top level state feature description.");
        }

        if (!interface.predictedfeaturename().empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Multi-function model must not use top level predictedFeatureName field.");
        }

        if (!interface.predictedprobabilitiesname().empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Multi-function model must not use top level predictedProbabilitiesName field.");
        }

        if (!interface.traininginput().empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Multi-function model must not use top level training input feature description.");
        }

        return Result();
    }

    Result validateMultiFunctionDescription(const Specification::ModelDescription& interface, int modelVersion, const ValidationPolicy& validationPolicy) {
        if (!validationPolicy.allowsMultipleFunctions) {
            return  Result(ResultType::MODEL_TYPE_DOES_NOT_SUPPORT_MULTI_FUNCTION,
                           "This model type doesn't support multi-function syntax.");
        }

        if (modelVersion < MLMODEL_SPECIFICATION_VERSION_IOS18) {
            return  Result(ResultType::INVALID_COMPATIBILITY_VERSION,
                           "Multi-function syntax is only valid in specification verison >= " + std::to_string(MLMODEL_SPECIFICATION_VERSION_IOS18)+
                           ". This model has version " + std::to_string(modelVersion));
        }

        const auto& functions = interface.functions();
        auto functionNames = std::vector<std::string>();
        for (const auto& function: functions) {
            Result r = validateFeatureDescriptions(function, modelVersion, validationPolicy);
            if (!r.good()) {
                return r;
            }
            functionNames.push_back(function.name());
        }

        // The default function name must be in function name list.
        const auto& defaultFunctionName = interface.defaultfunctionname();
        if (find(functionNames.begin(), functionNames.end(), defaultFunctionName) == functionNames.end()) {
            return  Result(ResultType::INVALID_DEFAULT_FUNCTION_NAME,
                           "The default function name '" + defaultFunctionName + "' is not found in the function name list: " + componentsJoinedBy(functionNames, ","));
        }

        return Result();
    }

    Result validateModelDescription(const Specification::ModelDescription& interface, int modelVersion, const ValidationPolicy& validationPolicy) {
        Result result;
        if (interface.functions_size() > 0 || !interface.defaultfunctionname().empty()) {
            // The model uses multi-function configuration.

            // Validate it doesn't use top level feature descriptions
            result = validateModelLevelFeatureDescriptionsAreEmpty(interface);
            if (!result.good()) {
                return result;
            }

            result = validateMultiFunctionDescription(interface, modelVersion, validationPolicy);
            if (!result.good()) {
                return result;
            }
        } else {
            // The model doesn't use multi-function configuration.
            Result result = validateFeatureDescriptions(interface, modelVersion, validationPolicy);
            if (!result.good()) {
                return result;
            }
        }

        return result;
    }


    Result validateRegressorInterface(const Specification::ModelDescription& description, int modelVersion, const ValidationPolicy& validationPolicy) {

        if (description.predictedfeaturename() == "") {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Specification is missing regressor predictedFeatureName.");
        }

        // Validate feature descriptions
        Result result = validateFeatureDescriptions(description, modelVersion, validationPolicy);
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

    template <class Description>
    inline Result validateClassifierFeatureDescriptions_(const Description& interface,
                                                         bool expected_class_is_int64) {

        const auto& predictedFeatureName = interface.predictedfeaturename();
        const auto& probOutputName = interface.predictedprobabilitiesname();

        if (predictedFeatureName.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          "Specification is missing classifier predictedFeatureName");
        } else {
            auto expected_class = (expected_class_is_int64
                                   ? Specification::FeatureType::TypeCase::kInt64Type
                                   : Specification::FeatureType::TypeCase::kStringType);

            auto result = validateDescriptionsContainFeatureWithNameAndType(interface.output(),
                                                                            predictedFeatureName,
                                                                            {expected_class});
            if (!result.good()) {
                return result;
            }
        }

        if (!probOutputName.empty()) {
            // TODO @znation: validate array length below
            // and value type (must be double? different for different classifiers?)
            // TODO Probability outputs are always dictionaries!
            auto result = validateDescriptionsContainFeatureWithNameAndType(interface.output(),
                                                                            probOutputName,
                                                                            {Specification::FeatureType::TypeCase::kMultiArrayType, // TODO ARRAY TYPE IS INVALID, REMOVE
                                                                            Specification::FeatureType::TypeCase::kDictionaryType});
            if (!result.good()) {
                return result;
            }
        }

        return Result();
    }

    Result validateClassifierFeatureDescriptions(const Specification::ModelDescription& interface,
                                                 bool expected_class_is_int64) {
        return validateClassifierFeatureDescriptions_(interface, expected_class_is_int64);
    }

    Result validateClassifierFeatureDescriptions(const Specification::FunctionDescription& interface,
                                                 bool expected_class_is_int64) {
        return validateClassifierFeatureDescriptions_(interface, expected_class_is_int64);
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

    inline Result validateDefaultOptionalValues(const Specification::Model& format) {
        // - Validate default optional values for NN Model that
        //   - Has default value set if input is optional and spec 5 model
        // - Error out if model is not Neural Network
        const Specification::ModelDescription& description = format.description();

        for (const auto& input : description.input()) {
            if (input.type().isoptional()) {
                switch (input.type().multiarraytype().defaultOptionalValue_case()) {
                    case CoreML::Specification::ArrayFeatureType::kDoubleDefaultValue:
                    case CoreML::Specification::ArrayFeatureType::kFloatDefaultValue:
                    case CoreML::Specification::ArrayFeatureType::kIntDefaultValue:
                        // Default value for optional inputs is applicable
                        // only for NeuralNetwork models with Spec 5 (iOS 14) onwards.
                        if (format.Type_case() != Specification::Model::kNeuralNetwork &&
                            format.Type_case() != Specification::Model::kNeuralNetworkRegressor &&
                            format.Type_case() != Specification::Model::kNeuralNetworkClassifier &&
                            format.Type_case() != Specification::Model::kSerializedModel &&
                            format.Type_case() != Specification::Model::kMlProgram) {
                            return Result(ResultType::INVALID_MODEL_PARAMETERS,
                                          "Default optional values are only allowed for neural networks.");
                        }
                        if (format.specificationversion() < MLMODEL_SPECIFICATION_VERSION_IOS14) {
                            return Result(ResultType::INVALID_MODEL_INTERFACE,
                                          "Default value for optional inputs is supported from specification 5 (iOS 14) onwards!");
                        }
                        break;
                    default:
                        break;
                }
            }
        }
        return Result();
    }

    inline Result validateOptionalNN(const Specification::ModelDescription& description) {
        // just need to check that not all inputs are optional
        bool hasNotOptional = false;
        for (const auto& input : description.input()) {
            if (input.type().Type_case() == Specification::FeatureType::kStateType) { // ignore optionality for State type input (which is always non-optional)
                hasNotOptional = true;
                continue;
            }
            if (!input.type().isoptional()) {
                hasNotOptional = true;
                break;
            }
        }
        if (description.input().size() > 0 && !hasNotOptional) {
            return Result(ResultType::INVALID_MODEL_PARAMETERS, "At least one feature for a neural network must NOT be optional.");
        }
        return Result();
    }

    Result validateOptional(const Specification::Model& format) {
        Result r;
        r = validateDefaultOptionalValues(format);

        if (!r.good()) {
            return r;
        }

        switch (format.Type_case()) {
            case Specification::Model::kImputer:
                // Imputed values can be handled by replacing a particular value, so
                // optional is not required.
                break;
            case Specification::Model::kNeuralNetwork:
            case Specification::Model::kNeuralNetworkRegressor:
            case Specification::Model::kNeuralNetworkClassifier:
            case Specification::Model::kSerializedModel:
            case Specification::Model::kMlProgram:
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
            case Specification::Model::kItemSimilarityRecommender:
                // allow arbitrary optional in the recommender.  The recommender valiadator catches these.
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

    Result validateCanModelBeUpdatable(const Specification::Model& format) {
        Result r;
        switch (format.Type_case()) {
            case Specification::Model::kNeuralNetwork:
            case Specification::Model::kNeuralNetworkRegressor:
            case Specification::Model::kNeuralNetworkClassifier:
            case Specification::Model::kKNearestNeighborsClassifier:
            case Specification::Model::kPipeline:
            case Specification::Model::kPipelineRegressor:
            case Specification::Model::kPipelineClassifier:
                return r;
            default: {
                std::string err;
                err = "This model type is not supported for on-device update.";
                return Result(ResultType::INVALID_UPDATABLE_MODEL_PARAMETERS, err);
            }
        }
    }
}
