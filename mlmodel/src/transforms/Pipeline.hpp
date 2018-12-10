//
//  Pipeline.hpp
//  libmlmodelspec
//
//  Created by Hoyt Koepke on 11/17/16.
//  Copyright © 2016 Apple. All rights reserved.
//

#ifndef Pipeline_hpp
#define Pipeline_hpp

#include <stdio.h>
#include "../Model.hpp"

namespace CoreML {

namespace Specification {
  class Pipeline;
}

namespace Model {
  void initPipeline(CoreML::Specification::Model* model, const std::string& description);
  void initPipelineRegressor(CoreML::Specification::Model* model,
                            const std::string& predictedValueOutputName,
                            const std::string& description);
  void initPipelineClassifier(CoreML::Specification::Model* model,
                             const std::string& predictedClassName,
                             const std::string& probabilityName,
                             const std::string& description);
  void addModelToPipeline(const Specification::Model& spec, Specification::Pipeline* pipeline);
} // namespace Model

} // namespace CoreML

#endif /* Pipeline_hpp */
