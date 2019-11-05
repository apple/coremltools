//
//  ProgramValidator.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 10/30/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ProgramValidator.hpp"
#include "ParameterValidator.hpp"

namespace CoreML {

    template <>
    Result validate<MLModelType_program>(const Specification::Model& format) {
        const auto& interface = format.description();

        // This isn't true for classifiers and regressors -- need to template specialize it to make these work
        if (!std::all_of(interface.output().begin(),
                         interface.output().end(),
                         [](const Specification::FeatureDescription& output) {
                             return output.type().Type_case() == Specification::FeatureType::kMultiArrayType ||
                             output.type().Type_case() == Specification::FeatureType::kImageType;
                         })) {
                             return Result(ResultType::INVALID_MODEL_INTERFACE,
                                           "Neural Network outputs must be either an image or MLMultiArray.");
                        }

        std::set<std::string> outputBlobNames;

        // TOOD: Write a NN Program validator
        Result r = Result();

        if (r.good()) {
            // Make sure that all of the model interface's outputs are actually produced by some blob
            for (const auto& output : format.description().output()) {
                
                const std::string& name = output.name();
                
                std::string err;
                if (outputBlobNames.count(name) == 0) {
                    err = "Interface specifies output '" + name + "' which is not produced by any layer in the neural network.";
                    return Result(ResultType::INVALID_MODEL_INTERFACE, err);
                }
                outputBlobNames.erase(name);
            }
        }
        return r;
    }
};
