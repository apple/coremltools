//
//  AudioFeaturePrintValidator.cpp
//  mlmodel
//
//  Created by Tao Jia on 3/29/21.
//  Copyright © 2018 Apple Inc. All rights reserved.
//

#include "Result.hpp"
#include "Validators.hpp"
#include "ValidatorUtils-inl.hpp"
#include "../build/format/Model.pb.h"

namespace CoreML {

    template <>
    Result validate<MLModelType_audioFeaturePrint>(const Specification::Model &format) {
        const auto &interface = format.description();

        // make sure model is a vision feature print
        if (!format.has_audiofeatureprint()) {
            return Result(ResultType::INVALID_MODEL_PARAMETERS, "Model not an audio feature print.");
        }

        Result result;

        // validate the inputs: only one input with multiarray type is allowed
        result = validateDescriptionsContainFeatureWithTypes(interface.input(), 1, {Specification::FeatureType::kMultiArrayType});
        if (!result.good()) {
            return result;
        }

        // other validate logics here
        const auto &audioFeaturePrint = format.audiofeatureprint();
        switch (audioFeaturePrint.AudioFeaturePrintType_case()) {
            case Specification::CoreMLModels::AudioFeaturePrint::kSound:
                if (audioFeaturePrint.sound().version() == Specification::CoreMLModels::AudioFeaturePrint_Sound_SoundVersion_SOUND_VERSION_INVALID) {
                    return Result(ResultType::INVALID_MODEL_PARAMETERS, "Version for sound is invalid");
                }

                if (audioFeaturePrint.sound().version() == Specification::CoreMLModels::AudioFeaturePrint_Sound_SoundVersion_SOUND_VERSION_1) {
                    // validate the outputs: only one output with multiarray type is allowed for version 1
                    result = validateDescriptionsContainFeatureWithTypes(interface.output(), 1, {Specification::FeatureType::kMultiArrayType});
                    if (!result.good()) {
                        return result;
                    }
                }
                break;
            case Specification::CoreMLModels::AudioFeaturePrint::AUDIOFEATUREPRINTTYPE_NOT_SET:
                return Result(ResultType::INVALID_MODEL_PARAMETERS, "Type for audio feature print not set");
        }

        return result;
    }

}
