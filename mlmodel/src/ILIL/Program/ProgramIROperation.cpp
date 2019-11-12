//
//  ProgramIROperation.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIROperation.hpp"

#include "Format.hpp"
#include "ILIL/Program/ProgramIRBlock.hpp"
#include "ILIL/Program/ProgramIROperatorTypeConverter.hpp"
#include "ILIL/Program/ProgramIRValue.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

namespace {

using namespace ::google;

class ProgramIROperationImpl : public ProgramIROperation {
public:
    using AttributesMap = std::unordered_map<std::string, std::unique_ptr<IRValue>>;
    using ProtoAttributesMap = protobuf::Map<std::string, SpecValue>;

    using InputsMap = std::unordered_map<std::string, std::string>;
    using ProtoInputsMap = protobuf::Map<std::string, std::string>;

    using ProtoBlockVec = protobuf::RepeatedPtrField<V5::Block>;
    using ProtoNamedValueTypeVec = protobuf::RepeatedPtrField<V5::NamedValueType>;

    ProgramIROperationImpl(const OperationSpec& operation, ScopePtr scope)
        : m_scope(scope)
        , m_name(operation.name())
        , m_type(ProgramIROperatorTypeConverter::Instance().GetType(operation.type()))
        , m_attributes(ParseAttributes(operation.attributes()))
        , m_inputsMap(ParseInputMap(operation.inputs()))
        , m_outputNames(ParseOutputNames(operation.outputs()))
        , m_blocks(ParseBlocks(operation.blocks(), scope))
    { }

    const IRValue& GetAttribute(const std::string& name) const override {
        return *m_attributes.at(name);
    }

    const IRBlockPtrVec& GetBlocks() const override {
        return m_blocks;
    }

    const std::string& GetInput(const std::string& param) const override {
        return m_inputsMap.at(param);
    }

    const std::string& GetName() const override {
        return m_name;
    }

    const StringVec& GetOutputNames() const override {
        return m_outputNames;
    }

    const IRScope& GetScope() const override {
        return *m_scope;
    }

    IROperatorType GetType() const override {
        return m_type;
    }

private:
    static AttributesMap ParseAttributes(const ProtoAttributesMap& specAttributes)
    {
        AttributesMap attributes;
        for (const auto& nameAndValue : specAttributes) {
            attributes[nameAndValue.first] = ProgramIRValue::Parse(nameAndValue.second);
        }
        return attributes;
    }

    static IRBlockPtrVec ParseBlocks(const ProtoBlockVec& specBlocks, ScopePtr scope) {
        IRBlockPtrVec blocks;
        blocks.reserve(static_cast<size_t>(specBlocks.size()));
        for (const auto& specBlock : specBlocks) {
            blocks.push_back(ProgramIRBlock::Parse(specBlock, scope));
        }
        return blocks;
    }

    static InputsMap ParseInputMap(const ProtoInputsMap& specInputs)
    {
        InputsMap inputs;
        for (const auto& paramAndArg : specInputs) {
            inputs[paramAndArg.first] = paramAndArg.second;
        }
        return inputs;
    }

    static StringVec ParseOutputNames(const ProtoNamedValueTypeVec& specOutputs) {
        StringVec outputNames;
        outputNames.reserve(static_cast<size_t>(specOutputs.size()));
        for (const auto& output : specOutputs) {
            outputNames.push_back(output.name());
        }
        return outputNames;
    }

    ScopePtr m_scope;
    std::string m_name;
    IROperatorType m_type;
    AttributesMap m_attributes;
    StringVec m_inputArgs;
    InputsMap m_inputsMap;
    StringVec m_outputNames;
    IRBlockPtrVec m_blocks;
};
}

ProgramIROperation::~ProgramIROperation() = default;
ProgramIROperation::ProgramIROperation() = default;

/*static*/ std::unique_ptr<ProgramIROperation>
ProgramIROperation::Parse(const OperationSpec& operation, ScopePtr scope)
{
    return std::make_unique<ProgramIROperationImpl>(operation, std::move(scope));
}
