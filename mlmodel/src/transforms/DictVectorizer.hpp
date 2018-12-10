//
//  DictVectorizer.h
//  libmlmodelspec
//
//  Created by Hoyt Koepke on 2/7/17.
//  Copyright © 2017 Apple. All rights reserved.
//

#ifndef DictVectorizer_h
#define DictVectorizer_h

#include "../Result.hpp"
#include "../../build/format/OneHotEncoder_enums.h"

#include <string>

namespace CoreML {
    namespace Specification {
        class DictVectorizer;
    }
    namespace DictVectorizer {
        Result setFeatureEncoding(CoreML::Specification::DictVectorizer* dictVectorizer, const std::vector<int64_t>& container);
        Result setFeatureEncoding(CoreML::Specification::DictVectorizer* dictVectorizer, const std::vector<std::string>& container);
    }
}

#endif /* DictVectorizer_h */
