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

const IROperatorDescription& IROperation::GetDescription() const
{
    return GetIROperatorDescription(GetType());
}

IRScope::ConstIRValuePtr IROperation::GetValue(const std::string& name) const
{
    return GetScope().GetValue(name);
}
