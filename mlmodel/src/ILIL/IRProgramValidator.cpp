//  IRProgramValidator.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#include "ILIL/IRValueType.hpp"
#include "ILIL/IRBlock.hpp"
#include "ILIL/IROperation.hpp"
#include "ILIL/IRProgram.hpp"
#include "IRProgramValidator.hpp"
#include "ResultReason.hpp"
#include "ResultType.hpp"
#include "Context.hpp"

using namespace ::CoreML;
using namespace ::CoreML::ILIL;

#define CHECK_RESULT(x) { auto r = (x); if (!r.good()) { return r; } }

static Result ValidateOp(const IROperation& /*op*/, const Context& /*context*/);

static Result ValidateBlock(const IRBlock& block, const Context& context)
{
    // Validate arguments are defined in the surrounding scope
    auto parentScope = block.GetScope().GetParent();
    for (const auto& paramAndArg : block.GetInputs()) {
        if (paramAndArg.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::BLOCK_PARAM_NAME_EMPTY,
                          "Block parameter names must be non-empty.");
        }

        if (paramAndArg.second.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::BLOCK_ARG_NAME_EMPTY,
                          "Block argument names must be non-empty.");
        }

        if (parentScope && parentScope->TryGetType(paramAndArg.first)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::BLOCK_PARAM_NAME_SHADOWS,
                          "Block parameter '" + paramAndArg.second + "' shadows an earlier declaration.");
        }

        if (!parentScope || !parentScope->TryGetType(paramAndArg.second)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::BLOCK_ARG_VALUE_UNDEFINED,
                          "Undefined value '" + paramAndArg.second + "' used as block input.");
        }
    }

    for (const auto& outputName : block.GetOutputs()) {
        if (outputName.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::BLOCK_OUTPUT_NAME_EMPTY,
                          "Block output names must be non-empty.");
        }

        if (!block.GetScope().TryGetType(outputName)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::BLOCK_OUTPUT_VALUE_UNDEFINED,
                          "Undefined value '" + outputName + "' used as block output.");
        }
    }

    for (const auto& op : block.GetOperations()) {
        CHECK_RESULT(ValidateOp(*op, context));
    }

    return {};
}

static Result ValidateFunction(const IRProgram& program, const std::string& funcName, const Context& context)
{
    if (funcName.empty()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::FUNCTION_NAME_EMPTY,
                      "Function names must be non-empty.");
    }

    const auto& func = program.GetFunction(funcName);

    const auto& rootScope = program.GetScope();
    for (const auto& paramNameAndType : func.GetInputs()) {
        if (paramNameAndType.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::FUNCTION_PARAM_NAME_EMPTY,
                          "Function parameter names must be non-empty.");
        }

        if (rootScope.TryGetType(paramNameAndType.first)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::FUNCTION_PARAM_NAME_SHADOWS,
                          "Function parameter '" + paramNameAndType.first + "' shadows an earlier declaration.");
        }

        if (!paramNameAndType.second) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::FUNCTION_PARAM_TYPE_NULL,
                          "Function parameter types must be non-null.");
        }
    }

    if (func.GetOutputTypes().size() != func.GetBlock().GetOutputs().size()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::FUNCTION_BLOCK_MISMATCHED_RETURN_COUNT,
                      "A function must return the same number of values as its block.");
    }

    CHECK_RESULT(ValidateBlock(func.GetBlock(), context));

    for (size_t i = 0; i < func.GetOutputTypes().size(); ++i) {
        auto funcOutputType = func.GetOutputTypes().at(i);
        auto blockOutputName = func.GetBlock().GetOutputs().at(i);
        // We know the following will not throw because we've validated the values
        // named in the block's output are defined.
        const auto& blockOutputType = func.GetBlock().GetScope().GetType(blockOutputName);

        if (*funcOutputType != blockOutputType) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::FUNCTION_BLOCK_MISMATCHED_RETURN_TYPE,
                          "A function must return the same types as its block.");
        }
    }

    return {};
}

static bool SameDataTypes(Specification::ArrayFeatureType_ArrayDataType modelType, IRScalarValueTypeEnum programType)
{
    switch (modelType) {
        case Specification::ArrayFeatureType_ArrayDataType_DOUBLE:
            return (programType == IRScalarValueTypeEnum::Float64);
        case Specification::ArrayFeatureType_ArrayDataType_FLOAT32:
            return (programType == IRScalarValueTypeEnum::Float32);
        case Specification::ArrayFeatureType_ArrayDataType_INT32:
            return (programType == IRScalarValueTypeEnum::Int32);
        default:
            return false;
    }
}

static bool CompatibleShapes(const ::google::protobuf::RepeatedField<::google::protobuf::int64>& modelShapes,
                             const IRTensorValueType::Shape& programShapes)
{
    for (size_t i = 0; i < programShapes.size(); ++i) {
        auto dim = programShapes.at(i)->TryAs<IRConstantDimension>();
        if (!dim || static_cast<int64_t>(dim->GetSize()) != modelShapes.Get(static_cast<int>(i))) {
            return false;
        }

        // This must be a symbolic dimension, which matches all static shapes
    }

    return true;
}

static bool CompatibleShapes(const Specification::ArrayFeatureType_EnumeratedShapes& modelShapes,
                             const IRTensorValueType::Shape& programShapes,
                             bool isInput)
{
    // There's a bit of an impedence mismatch between Program and Model. Program does not
    // have a notion of enumerated shapes so the best we can do is validate that inputs
    // are compatible.
    if (isInput) {
        for (const auto& modelShape : modelShapes.shapes()) {
            for (size_t i = 0; i < programShapes.size(); ++i) {
                int64_t modelDim = modelShape.shape(static_cast<int>(i));
                auto constProgramDim = programShapes.at(i)->TryAs<IRConstantDimension>();
                if (constProgramDim) {
                    if (constProgramDim->GetSize() != static_cast<uint64_t>(modelDim)) {
                        return false;
                    }
                }

                // Must be a symbolic dimension, which will definitely match the model shape
            }
        }
    }

    return true;
}

static bool CompatibleDimensions(const Specification::SizeRange& modelRange,
                                 const IRDimension& programDim)
{
    // If the size range actually represents a single number, it can be represented by
    // a static length. Otherwise, the Program must have a symbol on that dimension.
    if (static_cast<int64_t>(modelRange.lowerbound()) == modelRange.upperbound()) {
        auto staticDim = programDim.TryAs<IRConstantDimension>();
        if (staticDim && staticDim->GetSize() == modelRange.lowerbound()) {
            return true;
        }
    }

    return programDim.Is<IRSymbolicDimension>();
}

static bool CompatibleShapes(const Specification::ArrayFeatureType_ShapeRange& modelShapes,
                             const IRTensorValueType::Shape& programShapes,
                             bool isInput)
{
    // Similar to above, Model provides more expressive constraints than Program.
    // Therefore, we only validate that the Program's inputs are at least as
    // permissive as the Model's constraints.
    if (isInput) {
        for (int i = 0; modelShapes.sizeranges_size(); ++i) {
            const auto& sizeRange = modelShapes.sizeranges(i);
            const auto& programDim = programShapes.at(static_cast<size_t>(i));

            if (!CompatibleDimensions(sizeRange, *programDim)) {
                return false;
            }
        }
    }

    return true;
}

static bool CompatibleShapes(const Specification::ArrayFeatureType& modelType,
                             const IRTensorValueType::Shape& shape,
                             bool isInput)
{
    switch (modelType.ShapeFlexibility_case()) {
        case Specification::ArrayFeatureType::kEnumeratedShapes:
            return CompatibleShapes(modelType.enumeratedshapes(), shape, isInput);
        case Specification::ArrayFeatureType::kShapeRange:
            return CompatibleShapes(modelType.shaperange(), shape, isInput);
        case Specification::ArrayFeatureType::SHAPEFLEXIBILITY_NOT_SET:
            break;
    }

    return CompatibleShapes(modelType.shape(), shape);
}

static bool CompatibleDimensions(uint64_t modelDim, const IRDimension& programDim)
{
    auto constantDim = programDim.As<IRConstantDimension>();
    if (constantDim) {
        return constantDim->GetSize() == modelDim;
    }

    return programDim.Is<IRSymbolicDimension>();
}

static bool CompatibleSizes(uint64_t modelNumChannels, uint64_t modelHeight, uint64_t modelWidth,
                            const IRTensorValueType::Shape& programShape)
{
    // NCHW --> (1, C, H, W)
    if (programShape.size() != 4) { return false; }
    if (!CompatibleDimensions(1, *programShape.at(0))) { return false; }
    if (!CompatibleDimensions(modelNumChannels, *programShape.at(1))) { return false; }
    if (!CompatibleDimensions(modelHeight, *programShape.at(2))) { return false; }
    if (!CompatibleDimensions(modelWidth, *programShape.at(3))) { return false; }

    return true;
}

static bool CompatibleSizes(const Specification::ImageFeatureType_ImageSize& modelSize,
                             uint64_t modelNumChannels,
                             const IRTensorValueType::Shape& programShape)
{
    return CompatibleSizes(modelNumChannels, static_cast<uint64_t>(modelSize.width()),
                           static_cast<uint64_t>(modelSize.height()), programShape);
}

static bool CompatibleSizes(const Specification::ImageFeatureType_EnumeratedImageSizes& modelSizes,
                            uint64_t modelNumChannels,
                            const IRTensorValueType::Shape& programShape,
                            bool isInput)
{
    // As above, Program does not have a notion of enumerated size so the best we can do is
    // validate that inputs are compatibile.
    if (isInput) {
        for (const auto& modelSize : modelSizes.sizes()) {
            if (!CompatibleSizes(modelSize, modelNumChannels, programShape)) {
                return false;
            }
        }
    }

    return true;
}

static bool CompatibleSizes(const Specification::ImageFeatureType_ImageSizeRange& modelSizes,
                            uint64_t modelNumChannels,
                            const IRTensorValueType::Shape& programShape,
                            bool isInput)
{
    // As above, Program does not have a notion of enumerated size so the best we can do is
    // validate that inputs are compatibile.
    if (isInput) {
        // NCHW --> (1, C, H, W)
        if (programShape.size() != 4) { return false; }
        if (!CompatibleDimensions(1, *programShape.at(0))) { return false; }
        if (!CompatibleDimensions(modelNumChannels, *programShape.at(1))) { return false; }
        if (!CompatibleDimensions(modelSizes.heightrange(), *programShape.at(2))) { return false; }
        if (!CompatibleDimensions(modelSizes.widthrange(), *programShape.at(3))) { return false; }
    }

    return true;
}

static bool CompatibleDimensions(const Specification::ImageFeatureType& modelType,
                                 uint64_t modelNumChannels,
                                 const IRTensorValueType::Shape& shape,
                                 bool isInput)
{
    switch (modelType.SizeFlexibility_case()) {
        case Specification::ImageFeatureType::kEnumeratedSizes:
            return CompatibleSizes(modelType.enumeratedsizes(), modelNumChannels, shape, isInput);
        case Specification::ImageFeatureType::kImageSizeRange:
            return CompatibleSizes(modelType.imagesizerange(), modelNumChannels, shape, isInput);
        case Specification::ImageFeatureType::SIZEFLEXIBILITY_NOT_SET:
            break;
    }

    return CompatibleSizes(modelNumChannels, static_cast<uint64_t>(modelType.height()),
                           static_cast<uint64_t>(modelType.width()), shape);
}

static Result SameTypes(const std::string& name,
                        const Specification::ArrayFeatureType& modelType,
                        const IRTensorValueType& programType,
                        bool isInput,
                        const std::string& whereSingular)
{
    auto rank = static_cast<size_t>(modelType.shape().size());
    if (programType.GetShape().size() != rank) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_INPUT_MISMATCHED_RANK
            : ResultReason::MODEL_MAIN_OUTPUT_MISMATCHED_RANK;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model input '" + name + "' has a different rank than its corresponding " + whereSingular + " to main.");
    }

    if (!SameDataTypes(modelType.datatype(), programType.GetScalarType())) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_MISMATCHED_INPUT_TYPE
            : ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_TYPE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model input '" + name + "' has a different type than its corresponding " + whereSingular + " to main.");
    }

    if (!CompatibleShapes(modelType, programType.GetShape(), isInput)) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_MISMATCHED_INPUT_SHAPE
            : ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_SHAPE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model input '" + name + "' has a different shape than its corresponding " + whereSingular + " to main.");
    }

    return {};
}

static Result SameTypes(const std::string& name,
                        const Specification::ImageFeatureType& modelType,
                        const IRTensorValueType& programType,
                        bool isInput,
                        const std::string& whereSingular)
{
    if (programType.GetScalarType() != IRScalarValueTypeEnum::Float32) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_BAD_IMAGE_INPUT_TYPE
            : ResultReason::MODEL_MAIN_BAD_IMAGE_OUTPUT_TYPE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "main input '" + name + "' must be a tensor of Float32.");
    }

    // Program images are NCHW, where C is 3 (RGB) or 1 (grayscale)
    uint64_t channels = modelType.colorspace() == CoreML::Specification::ImageFeatureType_ColorSpace_GRAYSCALE ? 1 : 3;
    auto imageTensorType = IRTensorValueType::Make(IRScalarValueTypeEnum::Float32, {
        IRConstantDimension::Make(1),
        IRConstantDimension::Make(channels),
        IRConstantDimension::Make(static_cast<uint64_t>(modelType.height())),
        IRConstantDimension::Make(static_cast<uint64_t>(modelType.width())) });

    if (modelType.height() == 0 || modelType.width() == 0) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_BAD_IMAGE_INPUT_SIZE
            : ResultReason::MODEL_MAIN_BAD_IMAGE_OUTPUT_SIZE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model image input '" + name + "' has empty height or width.");
    }

    if (!CompatibleDimensions(modelType, channels, programType.GetShape(), isInput)) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_MISMATCHED_INPUT_SHAPE
            : ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_SHAPE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model image input '" + name + "' has a different shape than its corresponding " + whereSingular + " to main.");
    }

    return {};
}

static Result SameTypes(const std::string& name,
                        const Specification::FeatureType& modelType,
                        const IRTensorValueType& programType,
                        bool isInput,
                        const std::string& whereSingular)
{
    switch (modelType.Type_case()) {
        case CoreML::Specification::FeatureType::kMultiArrayType:
            return SameTypes(name, modelType.multiarraytype(), programType, isInput, whereSingular);
        case CoreML::Specification::FeatureType::kImageType:
            return SameTypes(name, modelType.imagetype(), programType, isInput, whereSingular);
        default:
            // Hopefully this is unreachable since we've already validated our input and output types
            throw std::runtime_error("Unexpected input type.");
    }

    return {};
}

static Result CheckModelMainInputOutput(const Specification::FeatureType& modelType,
                                        const IRValueType* funcType,
                                        const std::string& modelValueName,
                                        bool isInput)
{
    std::string whereSingular = isInput ? "parameter" : "return value";
    std::string wherePlural = isInput ? "parameters" : "return values";

    if (!funcType) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::MODEL_MAIN_MISSING_INPUT_OUTPUT,
                      "Model main function missing " + whereSingular + " '" + modelValueName + "' specified by model.");
    }
    auto funcTensorType = funcType->TryAs<IRTensorValueType>();
    if (!funcTensorType) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::MODEL_MAIN_NON_TENSOR_INPUT_OUTPUT,
                      "Main function " + wherePlural + " must be tensors.");
    }

    return SameTypes(modelValueName, modelType, *funcTensorType, isInput, whereSingular);
}

/** Peform validation specific to the main entry point. */
static Result ValidateMain(const Specification::Model& model, const IRFunction& mainFunc)
{
    if (model.description().input_size() != static_cast<int>(mainFunc.GetInputs().size())) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::MODEL_MAIN_MISMATCHED_INPUT_COUNT,
                      "Model and main function must have same number of inputs.");
    }

    if (model.description().output_size() != static_cast<int>(mainFunc.GetOutputTypes().size())) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_COUNT,
                      "Model and main function must have same number of outputs.");
    }

    for (int i = 0; i < model.description().input_size(); ++i) {
        const auto& modelType = model.description().input(i).type();
        const auto& modelName = model.description().input(i).name();
        const auto& funcType = mainFunc.TryGetInputType(modelName);
        CHECK_RESULT(CheckModelMainInputOutput(modelType, funcType, modelName, /*isInput=*/ true));
    }

    for (int i = 0; i < model.description().output_size(); ++i) {
        const auto& modelType = model.description().output(i).type();
        const auto& modelName = model.description().output(i).name();
        const auto& funcType = mainFunc.GetOutputTypes().at(static_cast<size_t>(i));
        CHECK_RESULT(CheckModelMainInputOutput(modelType, funcType.get(), modelName, /*isInput=*/ false));
    }

    return {};
}


static void GetInputs(const IRBlock& block, std::unordered_set<std::string>& inputs);

static void GetInputs(const IROperation& op, std::unordered_set<std::string>& inputs)
{
    for (const auto& paramAndArg : op.GetInputs()) {
        inputs.insert(paramAndArg.second);
    }

    for (const auto& block : op.GetBlocks()) {
        GetInputs(*block, inputs);
    }
}

static void GetInputs(const IRBlock& block, std::unordered_set<std::string>& inputs)
{
    for (const auto& paramAndArg : block.GetInputs()) {
        inputs.insert(paramAndArg.second);
    }

    for (const auto& op : block.GetOperations()) {
        GetInputs(*op, inputs);
    }
}

static Result ValidateOp(const IROperation& op, const Context& context)
{
    if (op.GetName().empty()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_NAME_EMPTY,
                      "Operation names must be non-empty.");
    }
    
    auto opDescription = context.TryGetOperatorDescription(op);
    if (!opDescription) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_INVALID_IN_CONTEXT,
                      "Operation type " + op.GetType() + " does not exist in this context.");
    }
        
    // Validate inputs
    for (const auto& paramAndArg : op.GetInputs()) {
        if (paramAndArg.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_PARAM_NAME_EMPTY,
                          "Operation parameter names must be non-empty.");
        }

        if (paramAndArg.second.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_ARG_NAME_EMPTY,
                          "Operation argument names must be non-empty.");
        }

        if (op.GetScope().TryGetType(paramAndArg.second) == nullptr) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_ARG_VALUE_UNDEFINED,
                          "Undefined value '" + paramAndArg.second + "' used as operation input.");
        }
    }

    for (auto& nameAndTypeInfo: opDescription->GetExpectedInputs()) {
        auto input = op.TryGetInput(nameAndTypeInfo.first);
        if (input == nullptr) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_REQUIRED_ARG_NOT_FOUND,
                          "Input " + nameAndTypeInfo.first + " is missing for layer: " + op.GetName());
        }
        
        auto& inputType = op.GetScope().GetType(*input);
        if (!opDescription->IsValidType(nameAndTypeInfo.first, inputType)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_ARG_TYPE_MISMATCH,
                          "Input " + nameAndTypeInfo.first + " is incorrect type for layer: " + op.GetName());
        }
    }
    
    auto numExpectedInputs = opDescription->GetExpectedInputs().size();
    
    // TODO (rdar://59205123): remove 'numExpectedInputs != 0' qualifier after all ops support quick validation
    if (numExpectedInputs != 0 && op.GetNumInputs() > numExpectedInputs) {
        auto msg = "For operation of type '" + op.GetType() + "' and name '" + op.GetName() + "', " + \
                    "number of arguments must be: " + std::to_string(numExpectedInputs) + ". Provided " + std::to_string(op.GetNumInputs());
        return Result(ResultType::INVALID_MODEL_INTERFACE, ResultReason::OP_MISMATCHED_PARAM_COUNT, msg);
    }
    
    // Validate outputs
    auto parentScope = op.GetScope().GetParent();
    std::unordered_set<std::string> allInputs;
    GetInputs(op, allInputs);

    if (op.GetNumOutputs() < opDescription->GetMinOutputs() || op.GetNumOutputs() > opDescription->GetMaxOutputs()) {
        auto msg = "For operation of type '" + op.GetType() + "' and name '" + op.GetName() + "', " + \
                    "number of outputs must be within the range (inclusive): " + std::to_string(opDescription->GetMinOutputs()) + \
                   " : " + std::to_string(opDescription->GetMaxOutputs()) + ". Provided " + std::to_string(op.GetNumInputs());
        return Result(ResultType::INVALID_MODEL_INTERFACE, ResultReason::OP_MISMATCHED_OUTPUT_COUNT, msg);
    }

    for (const auto& outputName : op.GetOutputNames()) {
        if (outputName.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_OUTPUT_NAME_EMPTY,
                          "Operation output names must be non-empty.");
        }
        if (parentScope && parentScope->TryGetType(outputName)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_OUTPUT_NAME_SHADOWS,
                          "Operation output '" + outputName + "' shadows an earlier declaration.");
        }

        if (allInputs.count(outputName)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_ARG_OUTPUT_CIRCULAR_DEFINITION,
                          "Operation uses '" + outputName + "' as both an input and an output.");
        }
    }

    // Validate attributes
    for (const auto& nameAndValue : op.GetAttributes()) {
        if (nameAndValue.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_ATTRIBUTE_NAME_EMPTY,
                          "Attribute names must be non-empty.");
        }

        if (!nameAndValue.second) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_ATTRIBUTE_VALUE_UNDEFINED,
                          "Attribute values must be non-null.");
        }
    }

    // Parameter and output name check
    const std::unordered_set<std::string> outputNames(op.GetOutputNames().begin(), op.GetOutputNames().end());
    for (const auto& paramName : op.GetInputNames()) {
        if (outputNames.count(paramName)) {
            // Throwing Exception as Parse error should have raised for this case.
            throw std::runtime_error("Parameter and output names must not be the same.");
        }
    }

    CHECK_RESULT(opDescription->ValidateOp(op));

    // Validate any nested blocks
    for (const auto& block: op.GetBlocks()) {
        CHECK_RESULT(ValidateBlock(*block, context));
    }

    return {};
}

static Result ValidateParameters(const IRProgram& program)
{
    for (const auto& paramNameAndValue : program.GetParameters()) {
        if (paramNameAndValue.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::PARAMETER_NAME_EMPTY,
                          "Parameter names must be non-empty.");
        }

        if (!paramNameAndValue.second) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::PARAMETER_VALUE_UNDEFINED,
                          "Parameter values must be non-null.");
        }
    }

    return {};
}

static bool IsImagePreprocessingOp(const std::string& opType) {
    return opType == "scale_image";
}

static Result ValidatePreprocessingOp(const Specification::Model& model, const IROperation& op)
{
    const auto& arg = op.GetInput("x");
    auto modelInput = std::find_if(model.description().input().cbegin(),
                                   model.description().input().cend(),
                                   [&arg](const Specification::FeatureDescription& fd) {
        return fd.name() == arg; });

    if (modelInput == model.description().input().cend()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::MODEL_MAIN_MISSING_INPUT_OUTPUT,
                      "Input '" + arg + "' to image preprocessing operation '" + op.GetName() +
                      " ' must be a model input.");
    }

    if (!modelInput->type().has_imagetype()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::MODEL_INVALID_INPUT_TYPE,
                      "Input '" + arg + "' used by image preprocessing operation '" + op.GetName() +
                      "' must be an image.");
    }

    return {};
}

static Result ValidateStrayPreprocessing(const IRBlock& block, bool isMainBlock)
{
    for (const auto& op : block.GetOperations()) {
        if (IsImagePreprocessingOp(op->GetType()) && !isMainBlock) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_INVALID_IN_CONTEXT,
                          "Preprocessing operation '" + op->GetName() + "' may only appear in main block.");
        }

        for (const auto& subBlock : op->GetBlocks()) {
            CHECK_RESULT(ValidateStrayPreprocessing(*subBlock, /*isMainBlock=*/ false));
        }
    }

    return {};
}

static Result ValidatePreprocessing(const Specification::Model& model, const IRProgram& program)
{
    // (1) Validate inputs to preprocessing that are at least in the correct block.
    const auto& mainBlock = program.GetFunction("main").GetBlock();
    for (const auto& op : mainBlock.GetOperations()) {
        if (IsImagePreprocessingOp(op->GetType())) {
            CHECK_RESULT(ValidatePreprocessingOp(model, *op));
        }
    }

    // (2) Check for preprocessing ops in the wrong block.
    for (const auto& nameAndFunc : program.GetFunctions()) {
        CHECK_RESULT(ValidateStrayPreprocessing(nameAndFunc.second->GetBlock(), nameAndFunc.first == "main"));
    }

    return {};
}

Result IRProgramValidator::Validate(const Specification::Model& model, const IRProgram& program, const Context& context)
{
    CHECK_RESULT(ValidateParameters(program));

    const auto& functions = program.GetFunctions();

    { // Validate main()
        auto mainNameAndFunc = functions.find("main");
        if (mainNameAndFunc == functions.cend() || mainNameAndFunc->second == nullptr) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::PROGRAM_MISSING_MAIN_FUNCTION,
                          "Program does not include a 'main' function.");
        }
        CHECK_RESULT(ValidateMain(model, *mainNameAndFunc->second));
    }

    // Validate functions
    for (const auto& nameAndFunc : functions) {
        CHECK_RESULT(ValidateFunction(program, nameAndFunc.first, context));
    }

    // Validate input preprocessing ops
    CHECK_RESULT(ValidatePreprocessing(model, program));

    return {};
}
