//
//  ProgramIRValueType.hpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/9/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRValueType.hpp"

#include "Format.hpp"

namespace CoreML {
namespace ILIL {
namespace Program {

using SpecValueType = ::CoreML::Specification::V5::ValueType;

namespace ProgramIRValueType {

std::unique_ptr<IRValueType> Parse(const SpecValueType& type);

}
}
}
}
