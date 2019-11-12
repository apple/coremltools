//
//  ProgramIRBlock.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIRBlock.hpp"

#include "Format.hpp"
#include "ILIL/Program/ProgramIROperation.hpp"
#include "ILIL/Program/ProgramIRValue.hpp"
#include "ILIL/Program/ProgramIRValueType.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;

namespace {
using namespace ::google;

class ProgramIRBlockImpl : public ProgramIRBlock {
public:
    using InputMap = std::unordered_map<std::string, std::string>;
    using ScopePtr = std::shared_ptr<IRScope>;

    using SpecInputMap = protobuf::Map<std::string, std::string>;
    using SpecOutputVec = protobuf::RepeatedPtrField<std::string>;

    ProgramIRBlockImpl(const BlockSpec& block, ConstScopePtr parentScope)
        : m_scope(std::make_shared<IRScope>(parentScope))
        , m_inputs(ParseInputs(block.inputs()))
        , m_outputs(ParseOutputs(block.outputs()))
        , m_operations(ParseOperations(block, m_scope))
    {
        PopulateScope(block, *parentScope);
    }

    const std::string& GetArgumentName(const std::string& parameterName) override {
        return m_inputs.at(parameterName);
    }

    const ConstIROperationPtrVec& GetOperations() const override {
        return m_operations;
    }

    const StringVec& GetOutputs() const override {
        return m_outputs;
    }

    const IRScope& GetScope() const override {
        return *m_scope;
    }

private:
    static InputMap ParseInputs(const SpecInputMap& specInputs) {
        InputMap inputs;
        inputs.reserve(static_cast<size_t>(specInputs.size()));
        for (const auto& paramAndArg : specInputs) {
            inputs[paramAndArg.first] = paramAndArg.second;
        }
        return inputs;
    }

    static ConstIROperationPtrVec ParseOperations(const BlockSpec& block, ScopePtr scope)
    {
        std::vector<std::unique_ptr<IROperation>> operations;
        operations.reserve(static_cast<size_t>(block.outputs_size()));
        for (const auto& op : block.operations()) {
            operations.push_back(ProgramIROperation::Parse(op, scope));
        }
        return operations;
    }

    static StringVec ParseOutputs(const SpecOutputVec& specOutputs) {
        StringVec outputs;
        outputs.reserve(static_cast<size_t>(specOutputs.size()));
        for (const auto& specOutput : specOutputs) {
            outputs.push_back(specOutput);
        }
        return outputs;
    }

    /**
     Prepare the scope associated with this block.

     It's useful to recall that blocks are topologically sorted by
     convention, not necessity. Therefore, we need to collect types of
     all values bound by ops defined here. We also add entries for block
     parameters by copying info from the surrounding scope.
     */
    void PopulateScope(const BlockSpec& thisBlock, const IRScope& parentScope)
    {
        // Copy arguments into new scope
        for (const auto& paramAndArg : thisBlock.inputs()) {
            auto argType = parentScope.GetType(paramAndArg.second);
            m_scope->SetType(paramAndArg.first, std::move(argType));

            auto argValue = parentScope.GetValue(paramAndArg.second);
            if (argValue) {
                m_scope->SetValue(paramAndArg.first, std::move(argValue));
            }
        }

        // Add entries for ops within this block
        for (const auto& op : thisBlock.operations()) {
            for (const auto& output : op.outputs()) {
                auto outputType = ProgramIRValueType::Parse(output.type());
                m_scope->SetType(output.name(), std::move(outputType));

                for (const auto& attribute : op.attributes()) {
                    if (attribute.first == "value") {
                        auto outputValue = ProgramIRValue::Parse(attribute.second);
                        m_scope->SetValue(output.name(), std::move(outputValue));
                    }
                }
            }
        }
    }

    ScopePtr m_scope;
    InputMap m_inputs;
    StringVec m_outputs;
    ConstIROperationPtrVec m_operations;
};
}

ProgramIRBlock::~ProgramIRBlock() = default;
ProgramIRBlock::ProgramIRBlock() = default;

/*static*/ std::unique_ptr<ProgramIRBlock>
ProgramIRBlock::Parse(const BlockSpec& block, ConstScopePtr parentScope)
{
    return std::make_unique<ProgramIRBlockImpl>(block, std::move(parentScope));
}
