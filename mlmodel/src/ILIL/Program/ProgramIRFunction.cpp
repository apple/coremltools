//
//  ProgramIRFunction.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIRFunction.hpp"

#include "Format.hpp"
#include "ILIL/Program/ProgramIRBlock.hpp"
#include "ILIL/Program/ProgramIRValueType.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

namespace {

using namespace ::google;

class ProgramIRFunctionImpl : public ProgramIRFunction {
public:
    using ProtoNamedValueTypeVec = protobuf::RepeatedPtrField<V5::NamedValueType>;
    using ProtoValueTypeVec = protobuf::RepeatedPtrField<V5::ValueType>;

    ProgramIRFunctionImpl(ValueTypeMap&& inputs, ConstIRValueTypePtrVec&& outputs,
                          ConstIRScopePtr&& scope, ConstIRBlockPtr&& block)
        : m_inputs(std::move(inputs))
        , m_outputs(std::move(outputs))
        , m_scope(std::move(scope))
        , m_block(std::move(block))
    { }

    ProgramIRFunctionImpl(const FunctionSpec& function, ConstIRScopePtr parentScope)
        : m_inputs(ParseInputs(function.inputs()))
        , m_outputs(ParseOutputs(function.outputs()))
        , m_scope(ParseScope(function, parentScope))
        , m_block(ProgramIRBlock::Parse(function.block(), m_scope))
    { }

    const IRBlock& GetBlock() const override {
        return *m_block;
    }

    const ValueTypeMap& GetInputs() const override {
        return m_inputs;
    }

    const IRValueType& GetInputType(const std::string& paramName) const override {
        return *m_inputs.at(paramName);
    }

    const IRValueType* TryGetInputType(const std::string& paramName) const override {
        auto inputType = m_inputs.find(paramName);
        return inputType == m_inputs.cend() ? nullptr : inputType->second.get();
    }

    const ConstIRValueTypePtrVec& GetOutputTypes() const override {
        return m_outputs;
    }

    const IRScope& GetScope() const override {
        return *m_scope;
    }

    std::unique_ptr<const IRFunction> WithBlock(ConstIRBlockPtr block) const override {
        auto copy = std::make_unique<ProgramIRFunctionImpl>(*this);
        copy->m_block = std::move(block);
        return copy;
    }

    std::unique_ptr<const IRFunction> WithRenames(const RenameVec& renames) const override {
        auto copy = std::make_unique<ProgramIRFunctionImpl>(*this);

        // inputs
        for (const auto& oldAndNew : renames) {
            auto iter = copy->m_inputs.find(oldAndNew.first);
            if (iter != copy->m_inputs.end()) {
                copy->m_inputs[oldAndNew.second] = iter->second;
                copy->m_inputs.erase(oldAndNew.first);
            }
        }

        // scope
        copy->m_scope = copy->m_scope->WithRenames(renames);

        // block
        copy->m_block = copy->m_block->WithRenames(renames);

        return copy;
    }


private:
    static ValueTypeMap ParseInputs(const ProtoNamedValueTypeVec& specInputs) {
        ValueTypeMap inputs;
        inputs.reserve(static_cast<size_t>(specInputs.size()));
        for (const auto& specInput : specInputs) {
            inputs[specInput.name()] = ProgramIRValueType::Parse(specInput.type());
        }

        return inputs;
    }

    static ConstIRValueTypePtrVec ParseOutputs(const ProtoValueTypeVec& specOutputs) {
        ConstIRValueTypePtrVec outputs;
        outputs.reserve(static_cast<size_t>(specOutputs.size()));
        for (const auto& specOutput : specOutputs) {
            outputs.push_back(ProgramIRValueType::Parse(specOutput));
        }
        return outputs;
    }

    static std::unique_ptr<const IRScope> ParseScope(const FunctionSpec& function, ConstIRScopePtr parentScope)
    {
        auto scope = std::make_unique<IRScope>(parentScope);

        for (const auto& input : function.inputs()) {
            const auto& name = input.name();
            auto type = ProgramIRValueType::Parse(input.type());
            scope->SetType(name, std::move(type));
        }

        return scope;
    }

    ValueTypeMap m_inputs;
    ConstIRValueTypePtrVec m_outputs;
    ConstIRScopePtr m_scope;
    ConstIRBlockPtr m_block;
};
}

ProgramIRFunction::~ProgramIRFunction() = default;
ProgramIRFunction::ProgramIRFunction() = default;

/*static*/ std::unique_ptr<ProgramIRFunction>
ProgramIRFunction::Make(ValueTypeMap&& inputs, ConstIRValueTypePtrVec&& outputs,
                        ConstIRScopePtr scope, ConstIRBlockPtr&& block)
{
    return std::make_unique<ProgramIRFunctionImpl>(std::move(inputs), std::move(outputs),
                                                   std::move(scope), std::move(block));
}

/*static*/ std::unique_ptr<ProgramIRFunction>
ProgramIRFunction::Parse(const FunctionSpec& function, ConstIRScopePtr parentScope)
{
    return std::make_unique<ProgramIRFunctionImpl>(function, parentScope);
}
