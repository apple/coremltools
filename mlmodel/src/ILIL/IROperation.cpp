//
//  IROperation.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IROperation.hpp"

using namespace ::CoreML::ILIL;

IROperation::~IROperation() = default;

const IRValue& IROperation::GetValue(const std::string& name) const
{
    return GetScope().GetValue(name);
}

const IRValue* IROperation::TryGetValue(const std::string& name) const
{
    return GetScope().TryGetValue(name);
}
