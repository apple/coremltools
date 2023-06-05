//
//  ClassConfidenceThresholdingValidator.cpp
//  libmlmodelspec
#include "Result.hpp"
#include "Validators.hpp"
#include "ValidatorUtils-inl.hpp"
#include "../build/format/Model.pb.h"
#include <cmath>

namespace CoreML {

    template <>
    Result validate<MLModelType_classConfidenceThresholding>(const Specification::Model& format) {
        const auto& interface = format.description();
        Result result;

        result = validateModelDescription(interface, format.specificationversion());
        if (!result.good()) {
            return result;
        }

        // validate precisionRecallCurves
        google::protobuf::RepeatedPtrField<CoreML::Specification::PrecisionRecallCurve> precisionrecallcurves =
                format.classconfidencethresholding().precisionrecallcurves();
        int nCurves = precisionrecallcurves.size();
        if (nCurves > 0) {
            for (int i = 0; i < nCurves; ++i) {
                int precisionvaluesElts = precisionrecallcurves.Get(i).precisionvalues().vector().size();
                int precisionthreshElts = precisionrecallcurves.Get(i).precisionconfidencethresholds().vector().size();
                if (0 == precisionvaluesElts || precisionvaluesElts != precisionthreshElts) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS, "Zero length or mismatched precisionRecallCurves components");
                }
                
                int recallvaluesElts = precisionrecallcurves.Get(i).recallvalues().vector().size();
                int recallthreshElts = precisionrecallcurves.Get(i).recallconfidencethresholds().vector().size();
                if (0 == recallvaluesElts || recallvaluesElts != recallthreshElts) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS, "Zero length or mismatched precisionRecallCurves components");
                }
                            
                for (auto elt : precisionrecallcurves.Get(i).precisionvalues().vector()) {
                    if (std::isinf(elt) || std::isnan(elt) || elt < 0.0f) {
                        return Result(ResultType::INVALID_MODEL_PARAMETERS, "An element of precisionvalues is not a positive number or zero.");
                    }
                }
                
                for (auto elt : precisionrecallcurves.Get(i).precisionconfidencethresholds().vector()) {
                    if (std::isinf(elt) || std::isnan(elt) || elt < 0.0f) {
                        return Result(ResultType::INVALID_MODEL_PARAMETERS, "An element of precisionconfidencethresholds is not a positive number or zero.");
                    }
                }
                
                for (auto elt : precisionrecallcurves.Get(i).recallvalues().vector()) {
                    if (std::isinf(elt) || std::isnan(elt) || elt < 0.0f) {
                        return Result(ResultType::INVALID_MODEL_PARAMETERS, "An element of recallvalues is not a positive number or zero.");
                    }
                }
                
                for (auto elt : precisionrecallcurves.Get(i).recallconfidencethresholds().vector()) {
                    if (std::isinf(elt) || std::isnan(elt) || elt < 0.0f) {
                        return Result(ResultType::INVALID_MODEL_PARAMETERS, "An element of recallconfidencethresholds is not a positive number or zero.");
                    }
                }
            }
            return Result();
        } else {
            return Result(ResultType::INVALID_MODEL_PARAMETERS, "The ClassConfidenceThresholding model has no precisionRecallCurves.");
        }
    }
}

