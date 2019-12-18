//
//  ProgramValidator.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 10/30/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "Format.hpp"
#include "ILIL/Program/ProgramIRProgram.hpp"
#include "Result.hpp"
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
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Block parameter names must be non-empty.");
        }

        if (paramAndArg.second.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Block argument names must be non-empty.");
        }

        if (!parentScope || !parentScope->TryGetType(paramAndArg.second)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Undefined value '" + paramAndArg.second + "' used as block input.");
        }
    }

    for (const auto& outputName : block.GetOutputs()) {
        if (outputName.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Block output names must be non-empty.");
        }

        if (!block.GetScope().TryGetType(outputName)) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Undefined value '" + outputName + "' used as block output.");
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
        return Result(ResultType::INVALID_MODEL_INTERFACE, "Function names must be non-empty.");
    }

    const auto& func = program.GetFunction(funcName);

    for (const auto& paramNameAndType : func.GetInputs()) {
        if (paramNameAndType.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Function parameter names must be non-empty.");
        }

        if (!paramNameAndType.second) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Function parameter types must be non-null.");
        }
    }

    CHECK_RESULT(ValidateBlock(func.GetBlock()));

    return {};
}

/** Peform validation specific to the main entry point. */
static Result ValidateMain(const Specification::Model& /*model*/, const IRFunction& /*mainFunc*/)
{
    // TODO: validate model and main() have matching inputs and outputs (rdar://57895959)
    return {};
}

static Result ValidateOp(const IROperation& op)
{
    if (op.GetName().empty()) {
        return Result(ResultType::INVALID_MODEL_INTERFACE, "Operation names must be non-empty.");
    }

    // Validate inputs
    for (const auto& paramAndArg : op.GetInputs()) {
        if (paramAndArg.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Operation parameter names must be non-empty.");
        }

        if (paramAndArg.second.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Operation argument names must be non-empty.");
        }

        if (op.GetScope().TryGetType(paramAndArg.second) == nullptr) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Undefined value '" + paramAndArg.second + "' used as operation input.");
        }
    }

    // Validate outputs
    for (const auto& outputName : op.GetOutputNames()) {
        if (outputName.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Operation output names must be non-empty.");
        }
    }

    // Validate attributes
    for (const auto& nameAndValue : op.GetAttributes()) {
        if (nameAndValue.first.empty()) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Attribute names must be non-empty.");
        }

        if (!nameAndValue.second) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Attribute values must be non-null.");
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
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Parameter names must be non-empty.");
        }

        if (!paramNameAndValue.second) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Parameter values must be non-null.");
        }
    }

    return {};
}

template <>
Result CoreML::validate<MLModelType_program>(const Specification::Model& model)
{
    std::unique_ptr<IRProgram> program;
    try {
        program = ProgramIRProgram::Parse(model.program());
    }
    catch (std::exception& ex) {
        return Result(ResultType::INVALID_MODEL_INTERFACE, ex.what());
    }

    if (!program) {
        return Result(ResultType::INVALID_MODEL_INTERFACE, "Program is null.");
    }

    CHECK_RESULT(ValidateParameters(*program));

    const auto& functions = program->GetFunctions();

    { // Validate main()
        auto mainNameAndFunc = functions.find("main");
        if (mainNameAndFunc == functions.cend() || mainNameAndFunc->second == nullptr) {
            return Result(ResultType::INVALID_MODEL_INTERFACE, "Program does not include a 'main' function.");
        }
        CHECK_RESULT(ValidateMain(model, *mainNameAndFunc->second));
    }

    // Validate functions
    for (const auto& nameAndFunc : functions) {
        CHECK_RESULT(ValidateFunction(*program, nameAndFunc.first));
    }

    return {};
}
