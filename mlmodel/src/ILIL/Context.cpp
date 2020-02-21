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

#include <cassert>
#include <limits>
#include <memory>

using namespace ::CoreML::ILIL;
using UIntLimit = std::numeric_limits<uint64_t>;

static Context::IROpDescriptionMap DefineOps(Context::IROpDescriptionMap&& additionalOps) {
    // Helper functions
    using IROpDescriptionMap = Context::IROpDescriptionMap;
    auto InputMap = IROperatorDescription::MakeInputMap;
    auto TypeList = IROperatorInputDescription::MakeTypeList;

    // Type definitions
    // TODO (rdar://59478016) The frontend is still dependent on legacy scalartypes. They should be removed from this file
    //                          after the frontend refactor is completed.
    auto float32Scalar_legacy = IRScalarValueType::Float32();
    auto float32Scalar = IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Float32);
    auto float32Tensor = IRTensorValueType::Make(IRScalarValueTypeEnum::Float32, {IRConstantDimension::Make(0)}); // Can be any shape, just not rank 0

    auto int32Scalar_legacy = IRScalarValueType::Int32();
    auto int32Scalar = IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32);
    auto int32Tensor = IRTensorValueType::Make(IRScalarValueTypeEnum::Int32, {IRConstantDimension::Make(0)}); // Can be any shape, just not rank 0

    auto stringScalar_legacy = IRScalarValueType::String();
    auto stringScalar = IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::String);

    auto boolScalar_legacy = IRScalarValueType::Bool();
    auto boolScalar = IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool);

    /* Some basic input check routines

       Inputs are defined as one of the following:
            { name, { valid types } } // isConst -> False, isOptional -> False
            { name, { isConst, { valid types } } } // isOptional -> False
            { name, { isConst, isOptional, { valid types } } }

        Tensor shapes are only verified to be scalar (rank 0) or not scalar.
     
        NOTE: This file does not support checking more specific details than the above.
              More specialized validation should reside in the IROperationValidator.
    */
    auto noInputCheck = InputMap({});
    auto elementwiseUnaryInput = InputMap({ { "x", TypeList({ float32Tensor }) } });

    auto alphaInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "alpha",          { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
    });

    auto alphaBetaInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "alpha",          { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "beta",           { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
    });
    
    auto avgPoolInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "kernel_sizes",   { true,         TypeList({ int32Tensor   }) } },
        { "strides",        { true,         TypeList({ int32Tensor   }) } },
        { "pad_type",       { true,         TypeList({ stringScalar,  stringScalar_legacy  }) } },
        { "pad",            { true,  true,  TypeList({ int32Tensor   }) } },
        { "exclude_padding_from_average",
                            { true,         TypeList({ boolScalar,    boolScalar_legacy    }) } }
    });

    auto batchnormInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "mean",           { true,         TypeList({ float32Tensor }) } },
        { "variance",       { true,         TypeList({ float32Tensor }) } },
        { "gamma",          { true,         TypeList({ float32Tensor }) } },
        { "beta",           { true,         TypeList({ float32Tensor }) } },
        { "epsilon",        { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
    });

    auto condInput = InputMap({
        { "pred",                           TypeList({ float32Scalar, float32Scalar_legacy,
                                                       boolScalar,    boolScalar_legacy    })   },
    });

    auto convInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "W",                              TypeList({ float32Tensor })   },
        { "number_spatial_dimensions",
                            { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
        { "strides",        { true,         TypeList({ int32Tensor   }) } },
        { "pad_type",       { true,         TypeList({ stringScalar,  stringScalar_legacy  }) } },
        { "pad",            { true,  true,  TypeList({ float32Tensor }) } },
        { "dilations",      { true,         TypeList({ int32Tensor   }) } },
        { "group",          { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
        { "B",              { true,  false, TypeList({ float32Tensor }) } }
    });

    auto elementwiseBinaryInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "y",                              TypeList({ float32Tensor })   }
    });

    auto expandDimsInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "axis",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } }
    });
    
    auto fillInput = InputMap({
        {"shape",                           TypeList({ int32Tensor   })   },
        {"value",           { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
    });
    
    auto innerProductInput = InputMap({
        { "weights",                        TypeList({ float32Tensor })   },
        { "x",                              TypeList({ float32Tensor })   },
        { "bias",           { false, true,  TypeList({ float32Tensor }) } },
    });

    auto linearInput = InputMap({
        {"x",                               TypeList({ float32Tensor })   },
        {"weight",          { true,         TypeList({ float32Tensor }) } },
        {"bias",            { true,         TypeList({ float32Tensor }) } },
    });
    
    auto matmulInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "y",                              TypeList({ float32Tensor })   },
        { "transpose_x",    { true,         TypeList({ boolScalar,    boolScalar_legacy    }) } },
        { "transpose_y",    { true,         TypeList({ boolScalar,    boolScalar_legacy    }) } },
    });

    auto maxPoolInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "kernel_sizes",   { true,         TypeList({ int32Tensor   }) } },
        { "strides",        { true,         TypeList({ int32Tensor   }) } },
        { "pad_type",       { true,         TypeList({ stringScalar,  stringScalar_legacy  }) } },
        { "pad",            { true,  true,  TypeList({ int32Tensor   }) } },
    });

    auto padInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "pad",            { true,         TypeList({ int32Tensor   }) } },
        { "mode",           { true,         TypeList({ stringScalar,  stringScalar_legacy  }) } },
        { "constant_val",   { true,  false, TypeList({ float32Scalar, float32Scalar_legacy }) } },
    });

    auto preluInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "alpha",          { true,         TypeList({ float32Tensor }) } },
    });

    auto randomBernoulliInput = InputMap({
        { "shape",                          TypeList({ int32Tensor   })   },
        { "prob",           { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "seed",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
    });

    auto randomCategoricalInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "mode",           { true,         TypeList({ stringScalar,  stringScalar_legacy  }) } },
        { "size",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
        { "seed",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
    });

    auto randomNormalInput = InputMap({
        { "shape",                          TypeList({ int32Tensor   })   },
        { "mean",           { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "stddev",         { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "seed",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
    });
 
    auto randomUniformInput = InputMap({
        { "shape",                          TypeList({ int32Tensor   })   },
        { "low",            { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "high",           { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "seed",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
    });

    auto reduceMeanInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "axes",           { true,         TypeList({ int32Tensor   }) } },
        { "keep_dims",      { true,         TypeList({ boolScalar,    boolScalar_legacy    }) } },
    });

    auto reshapeInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "shape",                          TypeList({ int32Tensor   })   },
    });

    auto scaleImageInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "scale",          { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "blueBias",       { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "greenBias",      { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "redBias",        { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
        { "grayBias",       { true,         TypeList({ float32Scalar, float32Scalar_legacy }) } },
    });

    auto softplusParametricInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "alpha",          { true,         TypeList({ float32Tensor }) } },
        { "beta",           { true,         TypeList({ float32Tensor }) } },
    });

    auto softmaxInput = InputMap({
        { "logit",                          TypeList({ float32Tensor })   },
        { "axis",           { true,         TypeList({ int32Scalar,   int32Scalar_legacy   }) } },
    });

    auto squeezeInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "axes",           { true,         TypeList({ int32Tensor   }) } },
    });

    auto transposeInput = InputMap({
        { "x",                              TypeList({ float32Tensor })   },
        { "perm",           { true,         TypeList({ int32Tensor   }) } },
    });

    // TODO: <rdar://problem/58583272> NNV2: implement missing functionality for Reshape op
    // Input check for Shape Operator
    IROpDescriptionMap m = {
        { "abs", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "acos", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "add", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "asin", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "atan", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "avg_pool", IROperatorDescription(1, 1, avgPoolInput, &ValidatePooling) },
        { "batchnorm", IROperatorDescription(1, 1, batchnormInput, &ValidateBatchNorm) },
        { "ceil", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "clamped_relu", IROperatorDescription(1, 1, alphaBetaInput) },
        { "clip", IROperatorDescription(1, 1, alphaBetaInput) },
        { "cond", IROperatorDescription(1, UIntLimit::max(), condInput) },
        { "const", IROperatorDescription(1, 1, noInputCheck, &ValidateConst) },
        { "conv", IROperatorDescription(1, 1, convInput, &ValidateConv) },
        { "cos", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "cosh", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "elu", IROperatorDescription(1, 1, alphaInput) },
        { "equal", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "erf", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "exp", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "exp2", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "expand_dims", IROperatorDescription(1, 1, expandDimsInput, &ValidateExpandDims), },
        { "fill", IROperatorDescription(1, 1, fillInput) },
        { "floor", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "floor_div", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "greater", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "greater_equal", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "innerproduct", IROperatorDescription(1, 1, innerProductInput) },
        { "leaky_relu", IROperatorDescription(1, 1, alphaInput) },
        { "less", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "less_equal", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "linear", IROperatorDescription(1, 1, linearInput), },
        { "linear_activation", IROperatorDescription(1, 1, alphaBetaInput) },
        { "log", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "logical_and", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "logical_not", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "logical_or", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "logical_xor", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "matmul", IROperatorDescription(1, 1, matmulInput) },
        { "maximum", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "max_pool", IROperatorDescription(1, 1, maxPoolInput, &ValidatePooling) },
        { "minimum", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "mod", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "mul", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "not_equal", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "pad", IROperatorDescription(1, 1, padInput, &ValidatePad) },
        { "pow", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "prelu", IROperatorDescription(1, 1, preluInput, &ValidatePReLU) },
        { "random_bernoulli", IROperatorDescription(1, 1, randomBernoulliInput) },
        { "random_categorical", IROperatorDescription(1, 1, randomCategoricalInput) },
        { "random_normal", IROperatorDescription(1, 1, randomNormalInput) },
        { "random_uniform", IROperatorDescription(1, 1, randomUniformInput) },
        { "real_div", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "reducemean", IROperatorDescription(1, 1, reduceMeanInput) },
        { "relu", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "reshape", IROperatorDescription(1, 1, reshapeInput, &ValidateReshape) },
        { "round", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "rsqrt", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "scale_image", IROperatorDescription(1, 1, scaleImageInput) },
        { "scaled_tanh", IROperatorDescription(1, 1, alphaBetaInput) },
        { "sigmoid", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "sigmoid_hard", IROperatorDescription(1, 1, alphaBetaInput) },
        { "sign", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "sin", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "sinh", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "softmax", IROperatorDescription(1, 1, softmaxInput) },
        { "softplus", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "softplus_parametric", IROperatorDescription(1, 1, softplusParametricInput, &ValidateSoftplusParametric) },
        { "softsign", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "sqrt", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "squeeze", IROperatorDescription(1, 1, squeezeInput) },
        { "sub", IROperatorDescription(1, 1, elementwiseBinaryInput) },
        { "tan", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "tanh", IROperatorDescription(1, 1, elementwiseUnaryInput) },
        { "transpose", IROperatorDescription(1, 1, transposeInput) },
        { "threshold", IROperatorDescription(1, 1, alphaInput) },
        // Special Beta operator for validation of beta-operators
        { "beta", IROperatorDescription(0, UIntLimit::max(), noInputCheck)}
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
