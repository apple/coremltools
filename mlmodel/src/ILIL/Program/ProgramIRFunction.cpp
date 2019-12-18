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
    ConstIRScopePtr m_parentScope;
    ConstIRScopePtr m_scope;
    std::unique_ptr<ProgramIRBlock> m_block;
};
}

ProgramIRFunction::~ProgramIRFunction() = default;
ProgramIRFunction::ProgramIRFunction() = default;

/*static*/ std::unique_ptr<ProgramIRFunction>
ProgramIRFunction::Parse(const FunctionSpec& function, ConstIRScopePtr parentScope)
{
    return std::make_unique<ProgramIRFunctionImpl>(function, parentScope);
}
