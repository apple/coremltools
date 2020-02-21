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
 Helper methods
 */
static const IRTensorValueType& GetInputTensorType(const IROperation& op, const std::string& inputName) {
    return *op.GetScope().GetType(op.GetInput(inputName)).As<const IRTensorValueType>();
}
 

/**
 Validation methods for individual operators
 */
Result ValidateNoOp(const IROperation& /*op*/) {
    return {};
} // ValidateNoOp

Result ValidateBatchNorm(const IROperation& op) {
    const auto x_shape = GetInputTensorType(op, "x").GetShape();
    if (x_shape.size() != 3 && x_shape.size() != 4) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Attribute x must be rank 3 or 4 in layer: " + op.GetName());
    }

    auto xKnownDim = x_shape[1]->TryAs<const IRConstantDimension>();
    auto numChannels = xKnownDim ? xKnownDim->GetSize() : 0;

    for (std::string inputName : {"mean", "gamma", "beta"}) {
        const auto shape = GetInputTensorType(op, inputName).GetShape();
        if (shape.size() != 1) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "Input " + inputName + " must be rank 1 for layer: " + op.GetName());
        }

        auto inputKnownDim = shape[0]->As<const IRConstantDimension>();
        if (numChannels == 0) {
            numChannels = inputKnownDim->GetSize();
        } else if (numChannels != inputKnownDim->GetSize() ){
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "The size of dim 1 of x must be the length of dim 0 of " \
                          "mean, gamma, and beta. Dim sizes are inconsistent for layer: " + op.GetName());
        }
    }

    return Result();
} // ValidateBatchNorm

Result ValidateConst(const IROperation& op) {
    const auto val = op.TryGetAttribute("val");
    if (!val) return Result(ResultType::INVALID_MODEL_INTERFACE,
                            ResultReason::OP_PARAM_INVALID,
                            "Attribute val is undefined for const layer: " + op.GetName());
    return Result();
} // ValidateConst

Result ValidateConv(const IROperation& op) {
    // Verify x is correct rank
    auto& x_type = GetInputTensorType(op, "x");
    auto& x_shape = x_type.GetShape();
    if (x_shape.size() != 3 && x_shape.size() != 4) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Attribute x must be rank 3 or 4 in layer: " + op.GetName());
    }

    // Verify rank(w) == rank(x)
    auto& w_type =  GetInputTensorType(op, "W");
    auto& w_shape = w_type.GetShape();
    if (!x_type.IsVariadicRank() && !w_type.IsVariadicRank() && x_shape.size() != w_shape.size()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "W must be the a compatible dimensionality for x in layer: " + op.GetName());
    }

    // Verify num channels
    auto x_channel_dim = x_shape[1]->TryAs<const IRConstantDimension>();
    auto w_channel_dim = w_shape[1]->TryAs<const IRConstantDimension>();
    if (x_channel_dim) {
        if (w_channel_dim && w_channel_dim->GetSize() != x_channel_dim->GetSize()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "Number of input channels must be consistent between x and W for layer: " + op.GetName());
        }
    }

    // Verify spatial dimensions match
    auto num_dims = static_cast<size_t>(op.GetValue("number_spatial_dimensions").AsInt32());
    if ((!x_type.IsVariadicRank() && num_dims != x_shape.size() - 2) ||
        (!w_type.IsVariadicRank() && num_dims != w_shape.size() - 2)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "number_spatial_dimensions does not match dimensionality of x and W for layer: " + op.GetName());
    }

    // Verify padding selection is valid
    auto pad_type = op.GetValue("pad_type").AsString();
    if (pad_type != "valid" && pad_type != "same" && pad_type != "custom") {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Input pad type must be one of {valid, same, custom} for layer: " + op.GetName());
    }

    // Verify custom padding is provided if necessary
    auto pad_input = op.TryGetInput("pad");
    if (pad_type == "custom") {
        if (!pad_input) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "Layer " + op.GetName() + " chooses to use custom padding but does not provide custom padding.");
        }
        auto& padTensorType = GetInputTensorType(op, "pad");
        if (padTensorType.GetShape().size() != 1) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "Input pad should be rank 1 for layer: " + op.GetName());
        }
        auto padDim = padTensorType.GetShape()[0]->As<const IRConstantDimension>();
        if (padDim->GetSize() != 2 * num_dims) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_INVALID,
                          "The size of dim 0 of input pad must be (2 * # of spacial input dims) for layer: " + op.GetName());
        }
    } else if (pad_input) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Custom padding specified but custom padding not selected for layer: " + op.GetName());
    }

    // Verify dilations is shape [# spatial dimensions], rank 1
    auto& dilations_type = GetInputTensorType(op, "dilations");
    if (dilations_type.GetShape().size() != 1) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Input dialations should be rank 1 for layer: " + op.GetName());
    }
    auto dil_dim0 = dilations_type.GetShape()[0]->TryAs<const IRConstantDimension>();
    if (!dil_dim0 || dil_dim0->GetSize() != num_dims) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "The constant size of dim 0 of input dilations must be # of spacial input dims for layer: " + op.GetName());
    }

    // Verify # of input/output channels are divisible by group
    auto group = static_cast<size_t>(op.GetValue("group").AsInt32());
    auto chan_in = group;
    auto chan_out = group;
    if (x_channel_dim) {
        chan_in = x_channel_dim->GetSize();
    } else if (w_channel_dim) {
        chan_in = w_channel_dim->GetSize();
    }
    auto w_channel_out_dim = w_shape[0]->TryAs<const IRConstantDimension>();
    if (w_channel_out_dim) {
        chan_out = w_channel_out_dim->GetSize();
    }
    if (chan_out % group != 0 || chan_in % group != 0) {
    return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Both the number of input and output channels must be divisble by # of groups for layer: " + op.GetName());
    }

    // Verify bias is shape [# output channels], rank 1
    auto& bias_type = GetInputTensorType(op, "bias");
    if (bias_type.GetShape().size() != 1) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Input B should be rank 1 for layer: " + op.GetName());
    }
    auto bias_dim0 = bias_type.GetShape()[0]->TryAs<const IRConstantDimension>();
    if (!bias_dim0 || (w_channel_out_dim && bias_dim0->GetSize() != chan_out)) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "The constant size of dim 0 of input B must be # of output channels for layer: " + op.GetName());
    }

    return Result();
} // ValidateConv

Result ValidateExpandDims(const IROperation& op) {
    auto x_shape = GetInputTensorType(op, "x").GetShape();
    for (auto dim : x_shape) {
        // If any dims of x are variadic, we can't verify that axis is within range
        if (dim->IsVariadicRank()) return Result();
    }
    
    // Verify axis is valid for input shape
    auto num_dims = static_cast<int32_t>(x_shape.size());
    auto axis = op.GetValue(op.GetInput("axis")).AsInt32();
    if (axis > num_dims || axis < -num_dims - 1) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_PARAM_INVALID,
                      "Input 'axis' must be within range (-rank - 1, rank) for layer: " + op.GetName());
    }
    
    return Result();
} // ValidateExpandDims

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

Result ValidatePooling(const IROperation& op) {
    // TODO: rdar://59461139 (NNv2: More robust validations for Pooling ops)
    // kernel_sizes and strides must be length of 1 or same as rank of spatial dimensions
    const std::unordered_set<int> supportedSpatialRank = {1, 2};

    const auto* kernelSizesArgName = op.TryGetInput("kernel_sizes");
    if (kernelSizesArgName) {
        const auto& kernelSizesValue = op.GetValue(*kernelSizesArgName);
        const auto kernelSizesType = kernelSizesValue.GetType().TryAs<IRTensorValueType>();
        const auto kernelSizesRank = static_cast<int>(kernelSizesType->GetNumElements());
        if (supportedSpatialRank.find(kernelSizesRank) == supportedSpatialRank.end()) {
            auto err = "Invalid rank for kernel_sizes, should be length of 1 or same as spatial rank";
            return Result(ResultType::INVALID_MODEL_PARAMETERS,
                          ResultReason::OP_PARAM_INVALID, err);
        }
    }

    const auto* stridesArgName = op.TryGetInput("strides");
    if (stridesArgName) {
        const auto& stridesValue = op.GetValue(*stridesArgName);
        const auto stridesType = stridesValue.GetType().TryAs<IRTensorValueType>();
        const auto stridesRank = static_cast<int>(stridesType->GetNumElements());
        if (supportedSpatialRank.find(stridesRank) == supportedSpatialRank.end()) {
            auto err = "Invalid rank for strides, should be length of 1 or same as spatial rank";
            return Result(ResultType::INVALID_MODEL_PARAMETERS,
                          ResultReason::OP_PARAM_INVALID, err);
        }
    }

    // pad_type must be one of the following types
    const std::unordered_set<std::string> supportedPadTypes = {"same", "valid", "custom"};
    const auto* padTypeArgName = op.TryGetInput("pad_type");
    const auto padMode = op.GetValue(*padTypeArgName).AsString();
    if (supportedPadTypes.find(padMode) == supportedPadTypes.end()) {
        auto err = "Invalid pad_type '" + padMode + "'. Must be one of the ['same', 'valid', 'custom'].";
        return Result(ResultType::INVALID_MODEL_PARAMETERS,
                      ResultReason::OP_PARAM_INVALID, err);
    }

    int32_t pad[4];
    const auto* padArgName = op.TryGetInput("pad");
    if (padArgName) {
        const auto& padValue = op.GetValue(*padArgName);
        const auto padType = padValue.GetType().TryAs<IRTensorValueType>();
        const auto padRank = padType->GetNumElements();

        if (padRank != 2 && padRank != 4) {
            auto err = "Invalid rank for pad, should be length of 2 * spatial rank";
            return Result(ResultType::INVALID_MODEL_PARAMETERS,
                          ResultReason::OP_PARAM_INVALID, err);
        }

        padValue.CopyTo(pad, sizeof(int32_t) * padRank);

        // pad values must be all 0s for 'valid' pad_type
        if (padMode == "valid") {
            for (auto p : pad) {
                if (p != 0) {
                    auto err = "Pad values must be 0s for 'valid' pad type.";
                    return Result(ResultType::INVALID_MODEL_PARAMETERS,
                                  ResultReason::OP_PARAM_INVALID, err);
                }
            }
        }
    }

    return {};
} // ValidatePooling

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

} // namespace ILIL
} // namespace CoreML
