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
Result ValidateAdd(const IROperation& op);
Result ValidateBatchNorm(const IROperation& op);
Result ValidateClip(const IROperation& op);
Result ValidateCond(const IROperation& op);
Result ValidateConst(const IROperation& op);
Result ValidateConvolution(const IROperation& op);
Result ValidateElementwiseBinary(const IROperation& op);
Result ValidateElementwiseUnary(const IROperation& op);
Result ValidateExpandDims(const IROperation& op);
Result ValidateInnerProduct(const IROperation& op);
Result ValidateMatMul(const IROperation& op);
Result ValidateMaxPool(const IROperation& op);
Result ValidateNoOp(const IROperation& op);
Result ValidatePad(const IROperation& op);
Result ValidatePReLU(const IROperation& op);
Result ValidateRandomBernoulli(const IROperation& op);
Result ValidateRandomCategorical(const IROperation& op);
Result ValidateRandomNormal(const IROperation& op);
Result ValidateRandomUniform(const IROperation& op);
Result ValidateReduceMean(const IROperation& op);
Result ValidateReLU(const IROperation& op);
Result ValidateReshape(const IROperation& op);
Result ValidateSoftplusParametric(const IROperation& op);
Result ValidateSoftmax(const IROperation& op);
Result ValidateSqueeze(const IROperation& op);
Result ValidateThreshold(const IROperation& op);
Result ValidateTranspose(const IROperation& op);

} // namespace ILIL
} // namespace CoreML
