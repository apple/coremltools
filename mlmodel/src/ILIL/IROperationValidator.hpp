//
//  IROperationValidator.hpp
//  CoreML
//
//  Created by Bhushan Sonawane on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#pragma once

namespace CoreML {

class Result;

namespace ILIL {

class IROperation;

// Methods for Validation of individual operator
// TODO: <rdar://problem/58809118> [NNV2] Validators for Basic ops
Result ValidateNoOp(const IROperation& op); // Always returns successful result

Result ValidateBatchNorm(const IROperation& op);
Result ValidateConst(const IROperation& op);
Result ValidateConv(const IROperation& op);
Result ValidateExpandDims(const IROperation& op);
Result ValidatePad(const IROperation& op);
Result ValidatePooling(const IROperation& op);
Result ValidatePReLU(const IROperation& op);
Result ValidateReshape(const IROperation& op);
Result ValidateSoftplusParametric(const IROperation& op);

} // namespace ILIL
} // namespace CoreML
