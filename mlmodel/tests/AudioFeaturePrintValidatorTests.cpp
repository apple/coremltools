//
//  AudioFeaturePrintValidatorTests.cpp
//  CoreML_framework
//
//  Created by Tao Jia on 3/29/21.
//  Copyright © 2021 Apple Inc. All rights reserved.
//

#include "MLModelTests.hpp"
#include "../src/Format.hpp"
#include "../src/Model.hpp"
#include "ParameterTests.hpp"

#include "framework/TestUtils.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

using namespace CoreML;

int testAudioFeatureSoundPrintBasic() {

    Specification::ArrayFeatureType* inputArrayFeatureType = new Specification::ArrayFeatureType();
    Specification::FeatureType* inputFeatureType = new Specification::FeatureType();
    inputFeatureType->set_allocated_multiarraytype(inputArrayFeatureType);

    Specification::ArrayFeatureType* outputArrayFeatureType = new Specification::ArrayFeatureType();
    Specification::FeatureType* outputFeatureType = new Specification::FeatureType();
    outputFeatureType->set_allocated_multiarraytype(outputArrayFeatureType);

    Specification::ModelDescription* description = new Specification::ModelDescription();
    Specification::FeatureDescription* input = description->add_input();
    Specification::FeatureDescription* output = description->add_output();
    input->set_allocated_type(inputFeatureType);
    output->set_allocated_type(outputFeatureType);

    Specification::Model model;
    model.set_allocated_description(description);

    Result result = validate<MLModelType_audioFeaturePrint>(model);
    ML_ASSERT_BAD(result);

    auto *p = model.mutable_audiofeatureprint();
    result = validate<MLModelType_audioFeaturePrint>(model);
    ML_ASSERT_BAD(result);

    p->mutable_sound();
    result = validate<MLModelType_audioFeaturePrint>(model);
    ML_ASSERT_BAD(result);

    p->mutable_sound()->set_version(Specification::CoreMLModels::AudioFeaturePrint_Sound_SoundVersion_SOUND_VERSION_1);
    result = validate<MLModelType_audioFeaturePrint>(model);
    ML_ASSERT_GOOD(result);
    
    return 0;
}
