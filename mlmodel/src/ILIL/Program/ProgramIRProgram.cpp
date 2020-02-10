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
    using ProtoAttributesMap = protobuf::Map<std::string, V5::Value>;
    using ProtoFunctionMap = protobuf::Map<std::string, V5::Function>;
    using StringVec = std::vector<std::string>;

    ProgramIRProgramImpl(ParameterMap&& parameters, ConstIRScopePtr&& scope, IRFunctionMap&& functions)
        : m_parameters(std::move(parameters))
        , m_parameterNames(ParseParameterNames())
        , m_scope(std::move(scope))
        , m_functions(std::move(functions))
    { }

    ProgramIRProgramImpl(const ProgramSpec& program)
        : m_parameters(ParseParameters(program.parameters()))
        , m_parameterNames(ParseParameterNames())
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

    const IRValue* TryGetParameterValue(const std::string& name) const override {
        auto nameAndParamIter = m_parameters.find(name);
        return nameAndParamIter == m_parameters.cend()
            ? nullptr
        : nameAndParamIter->second.get();
    }

    const IRScope& GetScope() const override {
        return *m_scope;
    }

    std::unique_ptr<IRProgram> WithFunctions(IRFunctionMap&& funcs) const override {
        auto copy = std::make_unique<ProgramIRProgramImpl>(*this);
        copy->m_functions = std::move(funcs);
        return copy;
    }

    std::unique_ptr<IRProgram> WithRenames(const RenameVec& renames) const override {
        auto copy = std::make_unique<ProgramIRProgramImpl>(*this);

        // parameters
        for (const auto& oldAndNew : renames) {
            auto iter = copy->m_parameters.find(oldAndNew.first);
            if (iter != copy->m_parameters.end()) {
                copy->m_parameters[oldAndNew.second] = iter->second;
                copy->m_parameters.erase(oldAndNew.first);
            }
        }

        // parameter names
        for (auto& paramName : copy->m_parameterNames) {
            for (const auto& oldAndNew : renames) {
                if (paramName == oldAndNew.first) {
                    paramName.assign(oldAndNew.second);
                }
            }
        }

        // scope
        copy->m_scope = copy->m_scope->WithRenames(renames);

        // functions
        for (auto& nameAndFunc : copy->m_functions) {
            nameAndFunc.second = nameAndFunc.second->WithRenames(renames);
        }

        return copy;
    }
private:
    static IRFunctionMap ParseFunctions(const ProtoFunctionMap& specFunctions, ConstIRScopePtr thisScope) {
        IRFunctionMap functions;
        for (const auto& nameAndSpecFunc : specFunctions) {
            functions[nameAndSpecFunc.first] = ProgramIRFunction::Parse(nameAndSpecFunc.second, thisScope);
        }
        return functions;
    }

    StringVec ParseParameterNames() const {
        StringVec parameterNames;
        parameterNames.reserve(m_parameters.size());
        for (const auto& nameAndValue : m_parameters) {
            parameterNames.push_back(nameAndValue.first);
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

/*static*/ std::unique_ptr<ProgramIRProgram>
ProgramIRProgram::Make(ParameterMap&& parameters, ConstIRScopePtr scope, IRFunctionMap&& functions)
{
    return std::make_unique<ProgramIRProgramImpl>(std::move(parameters), std::move(scope), std::move(functions));
}

/*static*/ std::unique_ptr<ProgramIRProgram> ProgramIRProgram::Parse(const ProgramSpec& program)
{
    return std::make_unique<ProgramIRProgramImpl>(program);
}

