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
    using SpecInputMap = protobuf::Map<std::string, std::string>;
    using SpecOutputVec = protobuf::RepeatedPtrField<std::string>;

    ProgramIRBlockImpl(ScopePtr scope, InputBindingMap&& inputs, StringVec&& outputs, ConstIROperationPtrVec&& operations)
        : m_scope(std::move(scope))
        , m_inputs(std::move(inputs))
        , m_outputs(std::move(outputs))
        , m_operations(std::move(operations))
    { }

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

    const InputBindingMap& GetInputs() const override {
        return m_inputs;
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

    std::unique_ptr<IRBlock> WithOperations(ConstIROperationPtrVec&& ops) const override {
        auto copy = std::make_unique<ProgramIRBlockImpl>(*this);
        copy->m_operations = std::move(ops);
        return copy;
    }

    std::unique_ptr<IRBlock> WithRenames(const RenameVec& renames) const override {
        auto copy = std::make_unique<ProgramIRBlockImpl>(*this);

        // Input Params
        for (const auto& oldAndNew : renames) {
            auto iter = copy->m_inputs.find(oldAndNew.first);
            if (iter != copy->m_inputs.end()) {
                copy->m_inputs[oldAndNew.second] = iter->second;
                copy->m_inputs.erase(oldAndNew.first);
            }
        }

        // Input Args
        for (auto& inputParamAndName : copy->m_inputs) {
            for (const auto& oldAndNew : renames) {
                if (oldAndNew.first == inputParamAndName.second) {
                    inputParamAndName.second.assign(oldAndNew.second);
                }
            }
        }

        // Outputs
        for (auto& output : copy->m_outputs) {
            for (const auto& oldAndNew : renames) {
                if (output == oldAndNew.first) {
                    output.assign(oldAndNew.second);
                }
            }
        }

        // Scope
        copy->m_scope = copy->m_scope->WithRenames(renames);

        // Ops
        for (auto& op : copy->m_operations) {
            op = op->WithRenames(renames, copy->m_scope);
        }

        return copy;
    }

private:
    static InputBindingMap ParseInputs(const SpecInputMap& specInputs) {
        InputBindingMap inputs;
        inputs.reserve(static_cast<size_t>(specInputs.size()));
        for (const auto& paramAndArg : specInputs) {
            inputs[paramAndArg.first] = paramAndArg.second;
        }
        return inputs;
    }

    static ConstIROperationPtrVec ParseOperations(const BlockSpec& block, ScopePtr scope)
    {
        ConstIROperationPtrVec operations;
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
            auto argType = parentScope.GetTypeSharedPtr(paramAndArg.second);
            m_scope->SetType(paramAndArg.first, std::move(argType));

            auto argValue = parentScope.TryGetValueSharedPtr(paramAndArg.second);
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
                    if (attribute.first == "val") {
                        auto outputValue = ProgramIRValue::Parse(attribute.second);
                        m_scope->SetValue(output.name(), std::move(outputValue));
                    }
                }
            }
        }
    }

    ScopePtr m_scope;
    InputBindingMap m_inputs;
    StringVec m_outputs;
    ConstIROperationPtrVec m_operations;
};
}

ProgramIRBlock::~ProgramIRBlock() = default;
ProgramIRBlock::ProgramIRBlock() = default;

/*static*/ std::unique_ptr<ProgramIRBlock> ProgramIRBlock::Make(ScopePtr scope,
                                                                InputBindingMap&& inputs,
                                                                StringVec&& outputs,
                                                                ConstIROperationPtrVec&& operations)
{
    return std::make_unique<ProgramIRBlockImpl>(std::move(scope), std::move(inputs),
                                                std::move(outputs), std::move(operations));
}

/*static*/ std::unique_ptr<ProgramIRBlock>
ProgramIRBlock::Parse(const BlockSpec& block, ConstScopePtr parentScope)
{
    return std::make_unique<ProgramIRBlockImpl>(block, std::move(parentScope));
}
