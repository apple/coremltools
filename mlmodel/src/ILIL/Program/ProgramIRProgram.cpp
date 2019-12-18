//
//  ProgramIRProgram.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIRProgram.hpp"

#include "Format.hpp"
#include "ILIL/Program/ProgramIRFunction.hpp"
#include "ILIL/Program/ProgramIRValue.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

namespace {
using namespace ::google;

class ProgramIRProgramImpl : public ProgramIRProgram {
public:
    using ConstIRValuePtr = std::shared_ptr<const IRValue>;
    using ConstIRScopePtr = std::shared_ptr<const IRScope>;
    using ProtoAttributesMap = protobuf::Map<std::string, V5::Value>;
    using ProtoFunctionMap = protobuf::Map<std::string, V5::Function>;
    using StringVec = std::vector<std::string>;

    ProgramIRProgramImpl(const ProgramSpec& program)
        : m_parameters(ParseParameters(program.parameters()))
        , m_parameterNames(ParseParameterNames(program.parameters()))
        , m_scope(ParseScope(m_parameters))
        , m_functions(ParseFunctions(program.functions(), m_scope))
    {  }

    const IRFunction& GetFunction(const std::string& name) const override {
        auto nameAndFunctionIter = m_functions.find(name);
        if (nameAndFunctionIter == m_functions.cend()) {
            throw std::out_of_range("Function '" + name + "' does not exist.");
        }
        return *nameAndFunctionIter->second;
    }

    const IRFunctionMap& GetFunctions() const override {
        return m_functions;
    }

    const ParameterMap GetParameters() const override {
        return m_parameters;
    }

    const std::vector<std::string>& GetParameterNames() const override {
        return m_parameterNames;
    }

    const IRValue& GetParameterValue(const std::string& name) const override {
        auto nameAndParamIter = m_parameters.find(name);
        if (nameAndParamIter == m_parameters.cend()) {
            throw std::out_of_range("Parameter '" + name + "' does not exist.");
        }
        return *nameAndParamIter->second;
    }

    ConstIRValuePtr TryGetParameterValue(const std::string& name) const override {
        auto nameAndParamIter = m_parameters.find(name);
        return nameAndParamIter == m_parameters.cend()
            ? nullptr
            : nameAndParamIter->second;
    }

    const IRScope& GetScope() const override {
        return *m_scope;
    }

private:
    static IRFunctionMap ParseFunctions(const ProtoFunctionMap& specFunctions, ConstIRScopePtr thisScope) {
        IRFunctionMap functions;
        for (const auto& nameAndSpecFunc : specFunctions) {
            functions[nameAndSpecFunc.first] = ProgramIRFunction::Parse(nameAndSpecFunc.second, thisScope);
        }
        return functions;
    }

    static StringVec ParseParameterNames(const ProtoAttributesMap& specParameters) {
        StringVec parameterNames;
        parameterNames.reserve(specParameters.size());
        for (const auto& specNameAndValue : specParameters) {
            parameterNames.push_back(specNameAndValue.first);
        }
        std::sort(parameterNames.begin(), parameterNames.end());
        return parameterNames;
    }

    static ParameterMap ParseParameters(const ProtoAttributesMap& specParameters) {
        ParameterMap parameters;
        for (const auto& specNameAndValue : specParameters) {
            parameters[specNameAndValue.first] = ProgramIRValue::Parse(specNameAndValue.second);
        }
        return parameters;
    }

    /** Create entries in our scope for program-level parameters. */
    static ConstIRScopePtr ParseScope(const ParameterMap& parameters)
    {
        auto scope = std::make_unique<IRScope>(/*parentScope=*/ nullptr);
        for (const auto& nameAndValue : parameters) {
            scope->SetType(nameAndValue.first, nameAndValue.second->GetTypePtr());
            scope->SetValue(nameAndValue.first, nameAndValue.second);
        }
        return scope;
    }

    ParameterMap m_parameters;
    StringVec m_parameterNames;
    ConstIRScopePtr m_scope;
    IRFunctionMap m_functions;
};
}

ProgramIRProgram::~ProgramIRProgram() = default;
ProgramIRProgram::ProgramIRProgram() = default;

/*static*/ std::unique_ptr<ProgramIRProgram> ProgramIRProgram::Parse(const ProgramSpec& program)
{
    return std::make_unique<ProgramIRProgramImpl>(program);
}

