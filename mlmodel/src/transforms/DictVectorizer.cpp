//
//  DictVectorizer.cpp
//  libmlmodelspec
//
//  Created by Hoyt Koepke on 2/7/17.
//  Copyright © 2017 Apple. All rights reserved.
//

#include <stdio.h>
#include "DictVectorizer.hpp"
#include "../Format.hpp"
#include "../Model.hpp"

namespace CoreML {
namespace DictVectorizer {
    
    Result setFeatureEncoding(CoreML::Specification::DictVectorizer* dictVectorizer, const std::vector<int64_t>& container) {
        dictVectorizer->mutable_int64toindex()->clear_vector();
        
        for (auto element : container) {
            dictVectorizer->mutable_int64toindex()->add_vector(element);
        }
        return Result();
    }
    
    Result setFeatureEncoding(CoreML::Specification::DictVectorizer* dictVectorizer, const std::vector<std::string>& container) {
        dictVectorizer->mutable_stringtoindex()->clear_vector();
        
        for (auto element : container) {
            auto *value = dictVectorizer->mutable_stringtoindex()->add_vector();
            *value = element;
        }
        return Result();
    }
    
}}