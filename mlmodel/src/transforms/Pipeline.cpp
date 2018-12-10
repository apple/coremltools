//
//  Pipeline.cpp
//  libmlmodelspec
//
//  Created by Hoyt Koepke on 11/17/16.
//  Copyright © 2016 Apple. All rights reserved.
//

#include "Pipeline.hpp"
#include "../Format.hpp"

namespace CoreML {
namespace Model {
  static void initPipeline(CoreML::Specification::Model* model, const std::string& a, const std::string& b, const std::string& description, bool isClassifier) {
    initModel(model, description);
    auto* params = model->mutable_description();
    params->set_predictedfeaturename(a);
    if (isClassifier) {
      params->set_predictedprobabilitiesname(b);
      model->mutable_pipelineclassifier();
    } else {
      model->mutable_pipelineregressor();
    }
  }

  void initPipeline(CoreML::Specification::Model* model, const std::string& description) {
    initModel(model, description);
    model->mutable_pipeline();
  }

  void initPipelineRegressor(CoreML::Specification::Model* model,
              const std::string& predictedValueOutputName,
              const std::string& description) {
    initPipeline(model, predictedValueOutputName, "", description, false /* isClassifier */);
  }

  void initPipelineClassifier(CoreML::Specification::Model* model,
               const std::string& predictedClassName,
               const std::string& probabilityName,
               const std::string& description) {
    initPipeline(model, predictedClassName, probabilityName, description, true /* isClassifier */);
  }

  void addModelToPipeline(const Specification::Model& spec, Specification::Pipeline* pipeline) {
    google::protobuf::RepeatedPtrField< ::CoreML::Specification::Model >* container = pipeline->mutable_models();
    auto* contained = container->Add();
    *contained = spec;
  }

} // namespace Model
} // namespace CoreML
