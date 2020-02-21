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
#include "ILIL/Program/ProgramIRValue.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

namespace {

using namespace ::google;

class ProgramIROperationImpl : public ProgramIROperation {
public:
    using ProtoAttributesMap = protobuf::Map<std::string, SpecValue>;
    using ProtoBlockVec = protobuf::RepeatedPtrField<V5::Block>;
    using ProtoInputsMap = protobuf::Map<std::string, std::string>;
    using ProtoNamedValueTypeVec = protobuf::RepeatedPtrField<V5::NamedValueType>;

    ProgramIROperationImpl(ScopePtr scope,
                           const std::string& name,
                           const std::string& type,
                           AttributesMap&& attributes,
                           InputBindingMap&& inputs,
                           StringVec&& outputs,
                           IRBlockPtrVec&& blocks)
        : m_scope(std::move(scope))
        , m_name(name)
        , m_type(type)
        , m_attributes(std::move(attributes))
        , m_inputNames(ParseInputNames(inputs))
        , m_inputsMap(std::move(inputs))
        , m_outputNames(std::move(outputs))
        , m_blocks(std::move(blocks))
    { }

    ProgramIROperationImpl(const OperationSpec& operation, ScopePtr scope)
        : ProgramIROperationImpl(scope,
                                 operation.name(),
                                 operation.type(),
                                 ParseAttributes(operation.attributes()),
                                 ParseInputMap(operation.inputs()),
                                 ParseOutputNames(operation.outputs()),
                                 ParseBlocks(operation.blocks(), scope))
    { }

    const AttributesMap& GetAttributes() const override {
        return m_attributes;
    }

    const IRValue& GetAttribute(const std::string& name) const override {
        auto irvalueptr = TryGetAttribute(name);
        if (!irvalueptr) {
            throw std::out_of_range("Attribute does not exist.");
        }
        return *irvalueptr;
    }

    const IRValue* TryGetAttribute(const std::string& name) const override {
        auto nameAndValue = m_attributes.find(name);
        if (nameAndValue == m_attributes.cend()) return nullptr;
        return nameAndValue->second.get();
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

    const InputBindingMap& GetInputs() const override {
        return m_inputsMap;
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

    const std::string& GetType() const override {
        return m_type;
    }

    std::unique_ptr<IROperation> WithBlocks(IRBlockPtrVec&& blocks) const override
    {
        auto copy = std::make_unique<ProgramIROperationImpl>(*this);
        copy->m_blocks = std::move(blocks);
        return copy;
    }

    std::unique_ptr<IROperation> WithRenames(const RenameVec& renames, ScopePtr scope) const override
    {
        auto copy = std::make_unique<ProgramIROperationImpl>(*this);

        // Input arguments
        for (auto& inputParamAndName : copy->m_inputsMap) {
            for (const auto& oldAndNew : renames) {
                if (oldAndNew.first == inputParamAndName.second) {
                    inputParamAndName.second.assign(oldAndNew.second);
                }
            }
        }

        // Input names
        for (auto& input : copy->m_inputNames) {
            for (const auto& oldAndNew : renames) {
                if (oldAndNew.first == input) {
                    input.assign(oldAndNew.second);
                }
            }
        }

        // Outputs
        for (auto& output : copy->m_outputNames) {
            for (const auto& oldAndNew : renames) {
                if (oldAndNew.first == output) {
                    output.assign(oldAndNew.second);
                }
            }
        }

        // Blocks
        for (auto& block : copy->m_blocks) {
            block = block->WithRenames(renames);
        }

        copy->m_scope = std::move(scope);

        return copy;
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

    static InputBindingMap ParseInputMap(const ProtoInputsMap& specInputs)
    {
        InputBindingMap inputs;
        for (const auto& paramAndArg : specInputs) {
            inputs[paramAndArg.first] = paramAndArg.second;
        }
        return inputs;
    }

    static StringVec ParseInputNames(const InputBindingMap& inputs)
    {
        StringVec inputNames;
        inputNames.reserve(inputs.size());
        for (const auto& paramAndArg : inputs) {
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
    std::string m_type;
    AttributesMap m_attributes;
    StringVec m_inputNames;
    InputBindingMap m_inputsMap;
    StringVec m_outputNames;
    IRBlockPtrVec m_blocks;
};
}

ProgramIROperation::~ProgramIROperation() = default;
ProgramIROperation::ProgramIROperation() = default;

/*static*/ std::unique_ptr<ProgramIROperation>
ProgramIROperation::Make(ScopePtr scope, const std::string& name, const std::string& type,
                         AttributesMap&& attributes, InputBindingMap&& inputs,
                         StringVec&& outputs, IRBlockPtrVec&& blocks)
{
    return std::make_unique<ProgramIROperationImpl>(std::move(scope), name, type, std::move(attributes),
                                                    std::move(inputs), std::move(outputs), std::move(blocks));
}

/*static*/ std::unique_ptr<ProgramIROperation>
ProgramIROperation::Parse(const OperationSpec& operation, ScopePtr scope)
{
    return std::make_unique<ProgramIROperationImpl>(operation, std::move(scope));
}
