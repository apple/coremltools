//
//  IROperationValidator.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#include "IROperationValidator.hpp"
#include "IROperation.hpp"
#include "Result.hpp"
#include "ResultReason.hpp"
#include "ResultType.hpp"

#include "ILIL/IRValueType.hpp"

namespace CoreML {
namespace ILIL {

/**
 Validation methods for individual operators
 */
Result ValidateAdd(const IROperation& /*op*/) {
    return {};
} // ValidateAdd

Result ValidateBatchNorm(const IROperation& /*op*/) {
    return {};
} // ValidateBatchNorm

Result ValidateClip(const IROperation& /*op*/) {
    return {};
} // ValidateClip

Result ValidateCond(const IROperation& /*op*/) {
    return {};
} // ValidateConst

Result ValidateConst(const IROperation& /*op*/) {
    return {};
} // ValidateConst

Result ValidateConvolution(const IROperation& /*op*/) {
    return {};
} // ValidateConvolution

Result ValidateElementwiseBinary(const IROperation& /*op*/) {
    return {};
} // ValidateElementwiseBinary

Result ValidateElementwiseUnary(const IROperation& /*op*/) {
    return {};
} // ValidateElementwiseUnary

Result ValidateExpandDims(const IROperation& /*op*/) {
    return {};
} // ValidateConvolution

Result ValidateInnerProduct(const IROperation& /*op*/) {
    return {};
} // ValidateInnerProduct

Result ValidateMatMul(const IROperation& /*op*/) {
    return {};
} // ValidateMatMul

Result ValidateMaxPool(const IROperation& /*op*/) {
    return {};
} // ValidateMaxPool

Result ValidateNoOp(const IROperation& /*op*/) {
    return {};
} // ValidateNoOp

Result ValidatePad(const IROperation& op) {
    const auto* modeArgName = op.TryGetInput("mode");
    const auto mode = modeArgName ?
    op.GetValue(*modeArgName).AsString() : "constant";

    // Ensure mode == 'constant'
    std::unordered_set<std::string> supported_modes = {"constant", "reflect", "replicate"};
    if (supported_modes.count(mode) == 0) {
        std::string errorMessage = "pad supports three modes: `constant`, `reflect` and `reflection`. Given " + mode + ".\n";
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      errorMessage);
    }

    const auto& padArgName = op.GetInput("pad");
    const auto padValue = op.TryGetValue(padArgName);
    if (!padValue) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Padding values should be known.");
    }

    const auto padType = padValue->GetType().TryAs<IRTensorValueType>();
    if (!padType) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Padding type unknown.");
    }

    const auto padShape = padType->GetShape();
    if (padShape.size() != 2) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Padding must be rank 2.");
    }

    if (padShape[1]->As<IRConstantDimension>()->GetSize() != 2) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Padding must be of size <rank_of_input x 2> i.e. front and back padding should be specified.");
    }

    if (padShape[0]->As<IRConstantDimension>()->GetSize() > 2 && mode.compare("constant") != 0) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Padding for more than two dimension only supports `constant` mode!");
    }
    return {};
} // ValidatePad

Result ValidateRandomCategorical(const IROperation& /*op*/) {
    return {};
} // ValidateRandomCategorical

Result ValidateRandomBernoulli(const IROperation& /*op*/) {
    return {};
} // ValidateRandomBernoulli

Result ValidateRandomNormal(const IROperation& /*op*/) {
    return {};
} // ValidateRandomNormal

Result ValidateRandomUniform(const IROperation& /*op*/) {
    return {};
} // ValidateRandomUniform

Result ValidateReduceMean(const IROperation& /*op*/) {
    return {};
} // ValidateReduceMean

Result ValidateReLU(const IROperation& /*op*/) {
    return {};
} // ValidateReLU

Result ValidateReshape(const IROperation& op) {
    const auto& shapeArgName = op.GetInput("shape");
    const auto shapeValue = op.TryGetValue(shapeArgName);

    if (shapeValue) {
        const auto shapeType = shapeValue->GetType().As<IRTensorValueType>();
        const auto rank = shapeType->GetNumElements();

        if ((rank > 5) || (rank < 1)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "Rank of the shape parameter must be between 1 and 5 (inclusive).");
        }
    }
    return {};
} // ValidateReshape

Result ValidateSoftmax(const IROperation& /*op*/) {
    return {};
} // ValidateSoftmax

Result ValidateSqueeze(const IROperation& /*op*/) {
    return {};
} // ValidateSqueeze

Result ValidateThreshold(const IROperation& /*op*/) {
    return {};
} // ValidateThreshold

Result ValidateTranspose(const IROperation& /*op*/) {
    return {};
} // ValidateTranspose

} // namespace ILIL
} // namespace CoreML
