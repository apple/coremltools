//
//  FeatureVectorizer.hpp
//  libmlmodelspec
//
//  Created by Hoyt Koepke on 11/24/16.
//  Copyright © 2016 Apple. All rights reserved.
//

#ifndef FeatureVectorizer_hpp
#define FeatureVectorizer_hpp

#include "../Model.hpp"

namespace CoreML {
  namespace Specification {
    class FeatureVectorizer;
  }

  namespace FeatureVectorizer {
    Result add(CoreML::Specification::FeatureVectorizer* fv, const std::string& input_feature, size_t input_dimension);
    std::vector<std::pair<std::string, size_t> > get_inputs(const CoreML::Specification::FeatureVectorizer& fv);
  }
  
}


#endif /* FeatureVectorizer_hpp */
