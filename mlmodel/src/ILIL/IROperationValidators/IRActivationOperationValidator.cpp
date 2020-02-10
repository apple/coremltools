//
//  IROperationValidator.cpp
//  CoreML
//
//  Created by Kory Watson on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#include "ILIL/IROperationValidator.hpp"
#include "ILIL/IROperation.hpp"
#include "Result.hpp"
#include "ResultReason.hpp"
#include "ResultType.hpp"
#include "ILIL/IRValueType.hpp"

namespace CoreML {
namespace ILIL {
/** Verifies that a specific raw type is a tensor with shape [numInputChannels].
    If numInputChannels is 0, only verifies that the shape is a tensor with shape [some known number] */
static Result VerifyDimSizeIsNumChannels(const IRValueType& inputType, const IRValueType& weightType, const std::string& inputName, const std::string& weightName, const std::string& opname) {
    // Verify input is a tensor and that the weights tensor is shape [# of input channels]
    auto inputShape = inputType.As<const IRTensorValueType>()->GetShape();
    auto numInputDims = inputShape.size();
    if (numInputDims < 3) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_ARG_TYPE_MISMATCH,
                      "Input " + inputName + " should be at least rank 3 in layer: " + opname);

    }

    // Verify weights is flat (1 dimension)
    auto weightShape = weightType.As<const IRTensorValueType>()->GetShape();
    if (weightShape.size() != 1) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_ARG_TYPE_MISMATCH,
                      "Input " + weightName + " should be rank 1 in layer: " + opname);
    }
    auto weightDim0 = weightShape[0];
    if (!weightDim0->Is<IRConstantDimension>()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_ARG_TYPE_MISMATCH,
                      "Invalid " + weightName + " shape: rank should be known at compilation " +
                      "in layer:  '" + opname);
    }
    size_t weightDim0Size = weightDim0->As<IRConstantDimension>()->GetSize();

    // Verify input is a tensor and that the weights tensor is shape [# of input channels]
    // If the input dimension size is not known at compilation, assume the user knows what they're doing and don't check it...
    auto inputDim0 = inputShape[numInputDims - 3]->TryAs<IRConstantDimension>();
    if (inputDim0 != nullptr && inputDim0->GetSize() != weightDim0Size) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_ARG_TYPE_MISMATCH,
                      "Tensor " + weightName + " should be rank 1 with the same dim size as dimension '-3' " +
                      "in input " + inputName + " in layer:  '" + opname);
    }

    return Result();
}

Result ValidatePReLU(const IROperation& op) {
    // Set input names
    const std::string inputName = "x";
    const std::string weightName = "alpha";
    auto& inputType = op.GetScope().GetType(op.GetInput(inputName));
    auto& weightType = op.GetScope().GetType(op.GetInput(weightName));

    // Verify weight/input shapes
    return VerifyDimSizeIsNumChannels(inputType, weightType, inputName, weightName, op.GetName());
}

Result ValidateSoftplusParametric(const IROperation& op) {
    // Set input names
    const std::string inputName = "x";
    const std::string weightName = "alpha";
    const std::string weightName2 = "beta";
    auto& inputType = op.GetScope().GetType(op.GetInput(inputName));
    auto& weightType = op.GetScope().GetType(op.GetInput(weightName));
    auto& weightType2 = op.GetScope().GetType(op.GetInput(weightName2));

    // Verify weights are the same shape first
    if (weightType != weightType2) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_ARG_TYPE_MISMATCH,
                      "Inputs " + weightName + " and " + weightName2 + " should have the same scalar type and " +
                      "shape  in layer:  '" + op.GetName());
    }

    // Verify weight/input shapes
    Result r = VerifyDimSizeIsNumChannels(inputType, weightType, inputName, weightName, op.GetName());
    if (r.good()) r = VerifyDimSizeIsNumChannels(inputType, weightType2, inputName, weightName2, op.GetName());
    return r;
}

} // namespace ILIL
} // namespace CoreML
