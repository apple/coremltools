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
    using AttributesMap = std::unordered_map<std::string, std::shared_ptr<const IRValue>>;
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
        , m_inputNames(ParseInputNames(operation.inputs()))
        , m_inputsMap(ParseInputMap(operation.inputs()))
        , m_outputNames(ParseOutputNames(operation.outputs()))
        , m_blocks(ParseBlocks(operation.blocks(), scope))
    { }

    const IRValue& GetAttribute(const std::string& name) const override {
        auto nameAndValue = m_attributes.find(name);
        if (nameAndValue == m_attributes.cend()) {
            throw std::out_of_range("Attribute does not exist.");
        }
        return *nameAndValue->second;
    }

    const IRBlockPtrVec& GetBlocks() const override {
        return m_blocks;
    }

    const std::string& GetInput(const std::string& param) const override {
        auto paramAndArg = m_inputsMap.find(param);
        if (paramAndArg == m_inputsMap.cend()) {
            throw std::out_of_range("Parameter binding does not exist.");
        }
        return paramAndArg->second;
    }

    const std::string* TryGetInput(const std::string& param) const override {
        auto paramAndArgIter = m_inputsMap.find(param);
        return paramAndArgIter == m_inputsMap.cend() ? nullptr : &paramAndArgIter->second;
    }

    const StringVec& GetInputNames() const override {
        return m_inputNames;
    }

    const std::string& GetName() const override {
        return m_name;
    }

    uint64_t GetNumInputs() const override {
        return static_cast<uint64_t>(m_inputsMap.size());
    }

    uint64_t GetNumOutputs() const override {
        return static_cast<uint64_t>(m_outputNames.size());
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

    static StringVec ParseInputNames(const ProtoInputsMap& specInputs)
    {
        StringVec inputNames;
        inputNames.reserve(specInputs.size());
        for (const auto& paramAndArg : specInputs) {
            inputNames.push_back(paramAndArg.second);
        }
        std::sort(inputNames.begin(), inputNames.end());
        return inputNames;
    }

    static StringVec ParseOutputNames(const ProtoNamedValueTypeVec& specOutputs) {
        StringVec outputNames;
        outputNames.reserve(static_cast<size_t>(specOutputs.size()));
        for (const auto& output : specOutputs) {
            outputNames.push_back(output.name());
        }
        std::sort(outputNames.begin(), outputNames.end());
        return outputNames;
    }

    ScopePtr m_scope;
    std::string m_name;
    IROperatorType m_type;
    AttributesMap m_attributes;
    StringVec m_inputNames;
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
