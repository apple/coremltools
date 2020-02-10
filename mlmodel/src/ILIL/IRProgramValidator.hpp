//
//  IRProgramValidator.hpp
//  CoreML
//
//  Created by Bhushan Sonawane on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#pragma once

#include "Model.hpp"
#include "Result.hpp"

namespace CoreML {
namespace ILIL {

class Context;
class IRProgram;

namespace IRProgramValidator {
    Result Validate(const Specification::Model& model, const IRProgram& program, const Context& context);
} // namespace IRProgramValidator
} // namespace ILIL
} // namespace CoreML
