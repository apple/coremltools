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

        // TODO: Write a NN Program validator (rdar://57232966)
        Result r = Result();

        return r;
    }
};
