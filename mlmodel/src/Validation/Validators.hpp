//
//  Validator.hpp
//  libmlmodelspec
//
//  Created by Srikrishna Sridhar on 11/10/16.
//  Copyright © 2016 Apple. All rights reserved.
//

#ifndef Validator_h
#define Validator_h

#include "Format.hpp"
#include "Result.hpp"
#include "Globals.hpp"
#include "../../build/format/Model_enums.h"

#include "ValidatorUtils-inl.hpp"

namespace CoreML {

    namespace Specification {
        class Model;
        class ModelDescription;
        class Metadata;
        class Kernel;
    }

    /*
     * Template specialization of validation of the protobuf.
     *
     * @param  format Model spec format.
     * @return Result type of this operation.
     */
    template <MLModelType T> Result validate(const Specification::Model& format);

    struct ValidationPolicy {
        ValidationPolicy()
            :allowsEmptyInput(false),
             allowsEmptyOutput(false),
             allowsMultipleFunctions(false),
             allowsStatefulPrediction(false) {}

        /*
         * Initializes the policy based on the model type.
         */
        ValidationPolicy(MLModelType modelType);

        bool allowsEmptyInput;
        bool allowsEmptyOutput;
        bool allowsMultipleFunctions;
        bool allowsStatefulPrediction;
    };

    enum class FeatureIOType {
        INPUT,
        OUTPUT,
        STATE,
    };

    /*
     * Validate an individual feature description
     *
     * @param feature description
     # @param modelVersion The version of the model for backwards compatibility
     * @return Result type of this operation.
     */
    Result validateFeatureDescription(const Specification::FeatureDescription& desc, int modelVersion, FeatureIOType featureIOType = FeatureIOType::INPUT);

    /*
     * Validate model interface describes a valid transform
     *
     * @param  interface Model interface
     # @param modelVersion The version of the model for backwards compatibility
     * @param validationPolicy The validation policy.
     * @return Result type of this operation.
     */
    Result validateModelDescription(const Specification::ModelDescription& interface, int modelVersion, const ValidationPolicy& validationPolicy = ValidationPolicy());

    /*
     * Validate model interface describes a valid regressor
     *
     * @param  interface Model interface
     * @param validationPolicy The validation policy.
     * @return Result type of this operation.
     */
    Result validateRegressorInterface(const Specification::ModelDescription& interface, int modelVersion, const ValidationPolicy& validationPolicy = ValidationPolicy());


    /*
     * Validate classifier output feature descriptions of the model.
     */
    Result validateClassifierFeatureDescriptions(const Specification::ModelDescription& interface,
                                                 bool expected_class_is_int64);

    /*
     * Validate classifier output feature descriptions of a function model.
     */
    Result validateClassifierFeatureDescriptions(const Specification::FunctionDescription& interface,
                                                 bool expected_class_is_int64);

    /*
     * Validate feature descriptions in interface or function descriptions have supported names and type info.
     *
     * @param interface Model or Function interface.
     * @param modelVersion The version of the model for backwards compatibility.
     * @param validationPolicy The validation policy.
     * @return Result type of this operation.
     */
    template <class Description>
    inline Result validateFeatureDescriptions(const Description& interface, int modelVersion, const ValidationPolicy& validationPolicy) {
        if (interface.input_size() < 1) {
            if (!validationPolicy.allowsEmptyInput) {
                return Result(ResultType::MODEL_TYPE_DOES_NOT_SUPPORT_EMPTY_INPUT, "Models must have one or more inputs.");
            }

            if (modelVersion < MLMODEL_SPECIFICATION_VERSION_IOS18) {
                return  Result(ResultType::INVALID_COMPATIBILITY_VERSION,
                               "Empty input is only valid in specification verison >= " + std::to_string(MLMODEL_SPECIFICATION_VERSION_IOS18)+
                               ". This model has version " + std::to_string(modelVersion));
            }
        }

        if (!validationPolicy.allowsEmptyOutput && interface.output_size() < 1) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Models must have one or more outputs.");
        }

        for (const auto& input : interface.input()) {
            Result r = validateFeatureDescription(input, modelVersion, FeatureIOType::INPUT);
            if (!r.good()) { return r; }
        }

        for (const auto& output : interface.output()) {
            Result r = validateFeatureDescription(output, modelVersion, FeatureIOType::OUTPUT);
            if (!r.good()) { return r; }
        }

        for (const auto& state : interface.state()) {
            Result r = validateFeatureDescription(state, modelVersion, FeatureIOType::STATE);
            if (!r.good()) { return r; }
        }

        // If we got here, all inputs/outputs seem good independently of each other.
        return Result();
    }

    /*
     * Validate model interface describes a valid classifier
     *
     * @param  interface Model interface
     * @return Result type of this operation.
     */
    template<typename T, typename U>
    Result validateClassifierInterface(const T& model,
                                       const U& modelParameters,
                                       const bool allowEmptyLabels = false,
                                       const bool defaultClassLabelIsInt64 = false,
                                       const ValidationPolicy& validationPolicy = ValidationPolicy()) {

        bool expected_class_is_int64;

        // validate class labels
        switch (modelParameters.ClassLabels_case()) {
            case U::kInt64ClassLabels:
                if (!allowEmptyLabels && modelParameters.int64classlabels().vector_size() == 0) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS,
                                  "Classifier declared to have Int64 class labels must provide labels.");
                }

                if(modelParameters.stringclasslabels().vector_size() != 0) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS,
                                  "Classifier declared with Int64 class labels must provide exclusively Int64 class labels.");
                }

                expected_class_is_int64 = true;

                break;

            case U::kStringClassLabels:
                if (!allowEmptyLabels && modelParameters.stringclasslabels().vector_size() == 0) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS,
                                  "Classifier declared to have String class labels must provide labels.");
                }

                if(modelParameters.int64classlabels().vector_size() != 0) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS,
                    "Classifier declared with String class labels must provide exclusively String class labels.");
                }

                expected_class_is_int64 = false;

                break;

            case U::CLASSLABELS_NOT_SET:
                if (!allowEmptyLabels) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS, "Classifier models must provide class labels.");
                }
                expected_class_is_int64 = defaultClassLabelIsInt64;
                break;
        }
        const Specification::ModelDescription& interface = model.description();

            // Validate feature descriptions
        Result result = validateFeatureDescriptions(interface, model.specificationversion(), validationPolicy);
        if (!result.good()) {
            return result;
        }

        return validateClassifierFeatureDescriptions(interface, expected_class_is_int64);
    }

    /*
     * Validate optional inputs/outputs.
     * For most models, optional is not allowed (all inputs/outputs required).
     * Some models have different behavior.
     */
    Result validateOptional(const Specification::Model& format);

    /*
     * Validate if the model type can be set to updatable.
     */
    Result validateCanModelBeUpdatable(const Specification::Model& format);

}
#endif /* Validator_h */
