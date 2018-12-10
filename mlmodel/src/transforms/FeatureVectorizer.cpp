//
//  FeatureVectorizer.cpp
//  libmlmodelspec
//
//  Created by Hoyt Koepke on 11/24/16.
//  Copyright © 2016 Apple. All rights reserved.
//

#include "FeatureVectorizer.hpp"
#include "../Format.hpp"


namespace CoreML {
namespace FeatureVectorizer {

  Result add(CoreML::Specification::FeatureVectorizer* fv, const std::string& input_feature, size_t input_dimension) {
    auto container = fv->mutable_inputlist();
    auto c = new Specification::FeatureVectorizer_InputColumn;
    c->set_inputcolumn(input_feature);
    c->set_inputdimensions(input_dimension);
    
    // Already allocated.
    container->AddAllocated(c);
    return Result();
  }
  
  std::vector<std::pair<std::string, size_t> > get_inputs(const CoreML::Specification::FeatureVectorizer& fv) {
    auto container = fv.inputlist();
    std::vector<std::pair<std::string, size_t> > out(static_cast<size_t>(container.size()));
    
    for(int i = 0; i < container.size(); ++i) {
      out[static_cast<size_t>(i)] = {container[i].inputcolumn(), container[i].inputdimensions()};
    }
    
    return out;
  }

}}
