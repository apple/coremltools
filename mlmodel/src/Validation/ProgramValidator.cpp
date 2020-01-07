//
//  ProgramValidator.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 10/30/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "Format.hpp"
#include "ILIL/IRValueType.hpp"
#include "ILIL/Program/ProgramIRProgram.hpp"
#include "Result.hpp"
#include "ResultReason.hpp"
#include "Validation/NeuralNetwork/NeuralNetworkValidatorUtils.hpp"
#include "Validation/Validators.hpp"

using namespace ::CoreML;
using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;

#define CHECK_RESULT(x) { auto r = (x); if (!r.good()) { return r; } }

static Result ValidateOp(const IROperation& /*op*/);

static Result ValidateBlock(const IRBlock& block)
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
        CHECK_RESULT(ValidateOp(*op));
    }

    return {};
}

static Result ValidateFunction(const IRProgram& program, const std::string& funcName)
{
    if (funcName.empty()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::FUNCTION_NAME_EMPTY,
                      "Function names must be non-empty.");
    }

    const auto& func = program.GetFunction(funcName);

    for (const auto& paramNameAndType : func.GetInputs()) {
        if (paramNameAndType.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::FUNCTION_PARAM_NAME_EMPTY,
                          "Function parameter names must be non-empty.");
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

    CHECK_RESULT(ValidateBlock(func.GetBlock()));

    for (size_t i = 0; i < func.GetOutputTypes().size(); ++i) {
        auto funcOutputType = func.GetOutputTypes().at(i);
        auto blockOutputName = func.GetBlock().GetOutputs().at(i);
        // We know the following will not throw because we've validated the values
        // named in the block's output are defined.
        auto blockOutputType = func.GetBlock().GetScope().GetType(blockOutputName);

        if (*funcOutputType != *blockOutputType) {
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

static bool CompatibleDimensions(const ::google::protobuf::RepeatedField<::google::protobuf::int64>& modelShapes,
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

static bool CompatibleDimensions(const Specification::ArrayFeatureType_EnumeratedShapes& modelShapes,
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

static bool CompatibleDimensions(const Specification::ArrayFeatureType_ShapeRange& modelShapes,
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

            // If the size range actually represents a single number, it can be represented by
            // a static length. Otherwise, the Program must have a symbol on that dimension.
            if (static_cast<int64_t>(sizeRange.lowerbound()) == sizeRange.upperbound()) {
                auto staticDim = programDim->TryAs<IRConstantDimension>();
                if (staticDim && staticDim->GetSize() == sizeRange.lowerbound()) {
                    continue;
                }
            }

            if (!programDim->Is<IRSymbolicDimension>()) {
                return false;
            }
        }
    }

    return true;
}

static bool CompatibleDimensions(const Specification::ArrayFeatureType& modelType,
                                 const IRTensorValueType::Shape& shape,
                                 bool isInput)
{
    switch (modelType.ShapeFlexibility_case()) {
        case Specification::ArrayFeatureType::kEnumeratedShapes:
            return CompatibleDimensions(modelType.enumeratedshapes(), shape, isInput);
        case Specification::ArrayFeatureType::kShapeRange:
            return CompatibleDimensions(modelType.shaperange(), shape, isInput);
        case Specification::ArrayFeatureType::SHAPEFLEXIBILITY_NOT_SET:
            break;
    }

    return CompatibleDimensions(modelType.shape(), shape);
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

    if (!SameDataTypes(modelType.datatype(), programType.GetScalarType().GetType())) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_MISMATCHED_INPUT_TYPE
            : ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_TYPE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model input '" + name + "' has a different type than its corresponding " + whereSingular + " to main.");
    }

    if (!CompatibleDimensions(modelType, programType.GetShape(), isInput)) {
        auto reason = isInput
            ? ResultReason::MODEL_MAIN_MISMATCHED_INPUT_SHAPE
            : ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_SHAPE;
        return Result(ResultType::INVALID_MODEL_INTERFACE, reason,
                      "Model input '" + name + "' has a different shape than its corresponding " + whereSingular + " to main.");
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
            // TODO: rdar://58059103 (Validate image inputs and outputs in NNv2 Models)
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

static Result ValidateOp(const IROperation& op)
{
    if (op.GetName().empty()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::OP_NAME_EMPTY,
                      "Operation names must be non-empty.");
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

    // Validate outputs
    for (const auto& outputName : op.GetOutputNames()) {
        if (outputName.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE,
                          ResultReason::OP_OUTPUT_NAME_EMPTY,
                          "Operation output names must be non-empty.");
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

    // TODO: op-specific validation (rdar://57896032)

    // Validate any nested blocks
    for (const auto& block: op.GetBlocks()) {
        CHECK_RESULT(ValidateBlock(*block));
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

template <>
Result CoreML::validate<MLModelType_program>(const Specification::Model& model)
{
    // First things first, does the model have sensible input and output types?
    CHECK_RESULT(validateInputOutputTypes(model.description().input(),
                                          ResultReason::MODEL_INVALID_INPUT_TYPE, "inputs"));
    CHECK_RESULT(validateInputOutputTypes(model.description().output(),
                                          ResultReason::MODEL_INVALID_OUTPUT_TYPE, "outputs"));

    for (const auto& input : model.description().input()) {
        if (input.type().Type_case() == Specification::FeatureType::kMultiArrayType) {
            CHECK_RESULT(validateNdMultiArrayInputType(input.type().multiarraytype()));
        }
    }

    std::unique_ptr<IRProgram> program;
    try {
        program = ProgramIRProgram::Parse(model.program());
    }
    catch (std::exception& ex) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::PROGRAM_PARSE_THREW,
                      ex.what());
    }

    if (!program) {
        return Result(ResultType::INVALID_MODEL_INTERFACE,
                      ResultReason::PROGRAM_NULL, "Program is null.");
    }

    CHECK_RESULT(ValidateParameters(*program));

    const auto& functions = program->GetFunctions();

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
        CHECK_RESULT(ValidateFunction(*program, nameAndFunc.first));
    }

    return {};
}
