//
//  ProgramIRValue.hpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRValue.hpp"

#include "Format.hpp"

namespace CoreML {
namespace ILIL {
namespace Program {

using SpecValue = ::CoreML::Specification::V5::Value;

namespace ProgramIRValue {

std::shared_ptr<const IRValue> Parse(const SpecValue& value);

}
}
}
}
