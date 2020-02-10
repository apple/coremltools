//
//  ProgramValidator.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 10/30/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "Format.hpp"
#include "ILIL/Context.hpp"
#include "ILIL/IRProgramValidator.hpp"
#include "ILIL/IRValueType.hpp"
#include "ILIL/Program/ProgramIRProgram.hpp"
#include "Result.hpp"
#include "ResultReason.hpp"
#include "Validation/NeuralNetwork/NeuralNetworkValidatorUtils.hpp"
#include "Validation/Validators.hpp"

using namespace ::CoreML;
using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;

template <>
Result CoreML::validate<MLModelType_program>(const Specification::Model& model)
{
    // First things first, does the model have sensible input and output types?
    HANDLE_RESULT_AND_RETURN_ON_ERROR(validateInputOutputTypes(model.description().input(),
                                                               ResultReason::MODEL_INVALID_INPUT_TYPE, "inputs"));
    HANDLE_RESULT_AND_RETURN_ON_ERROR(validateInputOutputTypes(model.description().output(),
                                                               ResultReason::MODEL_INVALID_OUTPUT_TYPE, "outputs"));

    for (const auto& input : model.description().input()) {
        if (input.type().Type_case() == Specification::FeatureType::kMultiArrayType) {
            HANDLE_RESULT_AND_RETURN_ON_ERROR(validateNdMultiArrayInputType(input.type().multiarraytype()));
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

    // TODO: <rdar://problem/58809381> [NNV2] perform validation at one place during compilation
    auto context = std::make_shared<Context>();
    return IRProgramValidator::Validate(model, *program, *context);
}
