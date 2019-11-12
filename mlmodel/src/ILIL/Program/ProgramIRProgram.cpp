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
    using IRScopePtr = std::unique_ptr<IRScope>;
    using ParameterMap = std::unordered_map<std::string, ConstIRValuePtr>;
    using ProtoAttributesMap = protobuf::Map<std::string, V5::Value>;
    using ProtoFunctionMap = protobuf::Map<std::string, V5::Function>;


    ProgramIRProgramImpl(const ProgramSpec& program)
        : m_functions(ParseFunctions(program.functions()))
        , m_parameters(ParseParameters(program.parameters()))
        , m_scope()
    {
        PopulateScope();
    }

    const IRFunctionMap& GetFunctions() const override {
        return m_functions;
    }

    const IRValue& GetParameterValue(const std::string& name) const override {
        return *m_parameters.at(name);
    }

    const IRScope& GetScope() const override {
        return *m_scope;
    }

private:
    static IRFunctionMap ParseFunctions(const ProtoFunctionMap& specFunctions) {
        IRFunctionMap functions;
        for (const auto& nameAndSpecFunc : specFunctions) {
            functions[nameAndSpecFunc.first] = ProgramIRFunction::Parse(nameAndSpecFunc.second);
        }
        return functions;
    }

    static ParameterMap ParseParameters(const ProtoAttributesMap& specParameters) {
        ParameterMap parameters;
        for (const auto& specNameAndValue : specParameters) {
            parameters[specNameAndValue.first] = ProgramIRValue::Parse(specNameAndValue.second);
        }
        return parameters;
    }

    /** Create entries in our scope for program-level parameters. */
    void PopulateScope()
    {
        for (const auto& nameAndValue : m_parameters) {
            m_scope->SetType(nameAndValue.first, nameAndValue.second->GetTypePtr());
            m_scope->SetValue(nameAndValue.first, nameAndValue.second);
        }
    }

    IRFunctionMap m_functions;
    ParameterMap m_parameters;
    IRScopePtr m_scope;
};
}

ProgramIRProgram::~ProgramIRProgram() = default;
ProgramIRProgram::ProgramIRProgram() = default;

/*static*/ std::unique_ptr<ProgramIRProgram> ProgramIRProgram::Parse(const ProgramSpec& program)
{
    return std::make_unique<ProgramIRProgramImpl>(program);
}

