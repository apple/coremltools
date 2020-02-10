//
//  Context.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#include "Context.hpp"
#include "Result.hpp"
#include "IROperationValidator.hpp"
#include "ILIL/IROperatorDescription.hpp"
#include "ILIL/IROperationValidator.hpp"
#include "ILIL/IRValueType.hpp"
#include "ILIL/IROperation.hpp"

#include <limits>
#include <memory>

using namespace ::CoreML::ILIL;
using UIntLimit = std::numeric_limits<uint64_t>;

static Context::IROpDescriptionMap DefineOps(Context::IROpDescriptionMap&& additionalOps) {
    using IROpDescriptionMap = Context::IROpDescriptionMap;
    const auto InputMap = IROperatorDescription::MakeInputMap;
    const auto TypeList = IROperatorDescription::MakeTypeList;

    // Type definitions
    auto float32Scalar = IRScalarValueType::Float32();
    auto float32Tensor = IRTensorValueType::Make(IRScalarValueType::Float32(), {IRConstantDimension::Make(0)});

    auto int32Scalar = IRScalarValueType::Int32();
    auto int32Tensor = IRTensorValueType::Make(IRScalarValueType::Int32(), {IRConstantDimension::Make(0)});
    auto stringScalar = IRScalarValueType::String();

    // Some basic input check routines
    // NOTE: more specifialized should be part of reside near validator
    auto noInputCheck = InputMap({});
    auto elementwiseUnaryInput = InputMap({{"x", TypeList({float32Tensor})}});

    auto alphaInput = InputMap({
        {"x", TypeList({float32Tensor}) },
        {"alpha", TypeList({float32Scalar}) },
    });

    auto elementwiseBinaryInput = InputMap({
        {"x", TypeList({float32Tensor}) },
        {"y", TypeList({float32Tensor}) }
    });

    auto alphaBetaInput = InputMap({
        {"x", TypeList({float32Tensor}) },
        {"alpha", TypeList({float32Scalar}) },
        {"beta", TypeList({float32Scalar}) },
    });

    auto preluInput = InputMap({
        {"x", TypeList({float32Tensor}) },
        {"alpha", TypeList({float32Tensor}) },
    });

    auto softplusParametricInput = InputMap({
        {"x", TypeList({float32Tensor}) },
        {"alpha", TypeList({float32Tensor}) },
        {"beta", TypeList({float32Tensor}) },
    });

    auto padInput = InputMap({
        {"x", TypeList({float32Tensor}) },
        {"pad", TypeList({int32Tensor}) },
        {"mode", TypeList({stringScalar}) },
        {"constant_val", TypeList({float32Scalar}) },
    });

    // TODO: <rdar://problem/58583272> NNV2: implement missing functionality for Reshape op
    // Input check for Shape Operator
    IROpDescriptionMap m = {
        { "abs", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "acos", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "add", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "asin", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "atan", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "batchnorm", IROperatorDescription(1, 1, noInputCheck, &ValidateBatchNorm) },
        { "ceil", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "clamped_relu", IROperatorDescription(1, 1, alphaBetaInput, &ValidateNoOp) },
        { "clip", IROperatorDescription(1, 1, alphaBetaInput, &ValidateClip) },
        { "cond", IROperatorDescription(1, UIntLimit::max(), noInputCheck, &ValidateCond) },
        { "const", IROperatorDescription(1, 1, noInputCheck, &ValidateConst) },
        { "conv", IROperatorDescription(1, 1, noInputCheck, &ValidateConvolution) },
        { "cos", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "cosh", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "elu", IROperatorDescription(1, 1, alphaInput, &ValidateNoOp) },
        { "equal", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "erf", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateNoOp) },
        { "exp", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "exp2", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "expand_dims", IROperatorDescription(1, 1, noInputCheck, &ValidateExpandDims), },
        { "floor", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "floor_div", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary), },
        { "greater", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary), },
        { "greater_equal", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary), },
        { "innerproduct", IROperatorDescription(1, 1, noInputCheck, &ValidateInnerProduct), },
        { "leaky_relu", IROperatorDescription(1, 1, alphaInput, &ValidateNoOp) },
        { "less", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "less_equal", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "linear_activation", IROperatorDescription(1, 1, alphaBetaInput, &ValidateNoOp) },
        { "log", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "logical_and", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "logical_not", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "logical_or", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "logical_xor", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "matmul", IROperatorDescription(1, 1, noInputCheck, &ValidateMatMul) },
        { "maximum", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "maxpool", IROperatorDescription(1, 1, noInputCheck, &ValidateMatMul) },
        { "minimum", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "mod", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "mul", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "not_equal", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "pad", IROperatorDescription(1, 1, padInput, &ValidatePad) },
        { "pow", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "prelu", IROperatorDescription(1, 1, preluInput, &ValidatePReLU) },
        { "random_bernoulli", IROperatorDescription(1, 1, noInputCheck, &ValidateRandomBernoulli) },
        { "random_categorical", IROperatorDescription(1, 1, noInputCheck, &ValidateRandomCategorical) },
        { "random_normal", IROperatorDescription(1, 1, noInputCheck, &ValidateRandomNormal) },
        { "random_uniform", IROperatorDescription(1, 1, noInputCheck, &ValidateRandomUniform) },
        { "real_div", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "reducemean", IROperatorDescription(1, 1, noInputCheck, &ValidateReduceMean) },
        { "relu", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateReLU) },
        { "reshape", IROperatorDescription(1, 1, noInputCheck, &ValidateReshape) },
        { "round", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "rsqrt", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "scale_image", IROperatorDescription(1, 1, noInputCheck, &ValidateNoOp) },
        { "scaled_tanh", IROperatorDescription(1, 1, alphaBetaInput, &ValidateNoOp) },
        { "sigmoid", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateNoOp) },
        { "sigmoid_hard", IROperatorDescription(1, 1, alphaBetaInput, &ValidateNoOp) },
        { "sign", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "sin", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "sinh", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "softmax", IROperatorDescription(1, 1, noInputCheck, &ValidateSoftmax) },
        { "softplus", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateNoOp) },
        { "softplus_parametric", IROperatorDescription(1, 1, softplusParametricInput, &ValidateSoftplusParametric) },
        { "softsign", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateNoOp) },
        { "sqrt", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "squeeze", IROperatorDescription(1, 1, noInputCheck, &ValidateSqueeze) },
        { "sub", IROperatorDescription(1, 1, elementwiseBinaryInput, &ValidateElementwiseBinary) },
        { "tan", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateElementwiseUnary) },
        { "tanh", IROperatorDescription(1, 1, elementwiseUnaryInput, &ValidateNoOp) },
        { "transpose", IROperatorDescription(1, 1, noInputCheck, &ValidateTranspose) },
        { "threshold", IROperatorDescription(1, 1, alphaInput, &ValidateElementwiseUnary) },
        // Special Beta operator for validation of beta-operators
        { "beta", IROperatorDescription(0, UIntLimit::max(), noInputCheck, &ValidateNoOp)}
    };

    m.insert(additionalOps.begin(), additionalOps.end());

    return m;
} // Context::DefineOps

Context::Context(IROpDescriptionMap&& additionalOps) : m_opDescriptors(DefineOps(std::move(additionalOps))) { }

Context::Context() : Context(IROpDescriptionMap()) { }

Context::~Context() = default;

const IROperatorDescription& Context::GetOperatorDescription(const IROperation& op) const {
    return GetOperatorDescription(op.GetType());
} // Context::GetOperatorDescription

const IROperatorDescription* Context::TryGetOperatorDescription(const IROperation& op) const {
    return TryGetOperatorDescription(op.GetType());
} // Context::TryGetOperatorDescription

const IROperatorDescription& Context::GetOperatorDescription(const std::string& opType) const {
    auto result = TryGetOperatorDescription(opType);
    if (result == nullptr) throw std::runtime_error("Operator Description not found!");
    return *result;
} // Context::GetOperatorDescription

const IROperatorDescription* Context::TryGetOperatorDescription(const std::string& opType) const {
    auto desc = m_opDescriptors.find(opType);
    if (desc != m_opDescriptors.cend()) {
        return &desc->second;
    }
    // Check if given operator is a `beta-operator`
    if ( opType.find("beta", 0, 4) != std::string::npos) {
        desc = m_opDescriptors.find("beta");
        assert(desc != m_opDescriptors.cend() && "Beta Operator Description not present");
        return &desc->second;
    }
    // Operator description not listed
    return nullptr;
} // Context::TryGetOperatorDescription
