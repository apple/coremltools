//
//  ProgramIROperatorTypeConverter.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIROperatorTypeConverter.hpp"

#include <memory>
#include <mutex>

using ::CoreML::ILIL::IROperatorType;
using ::CoreML::ILIL::Program::ProgramIROperatorTypeConverter;

ProgramIROperatorTypeConverter::~ProgramIROperatorTypeConverter() = default;

ProgramIROperatorTypeConverter::ProgramIROperatorTypeConverter()
    : m_nameToType(ProgramIROperatorTypeConverter::MakeNameToTypeMap())
{ }

/*static*/ const ProgramIROperatorTypeConverter& ProgramIROperatorTypeConverter::Instance()
{
#pragma clang diagnostic push

    // This warns us that tearing down "instance" will happen at program exit.
    // The concern is that this happens in an effectively non-deterministic
    // order which can cause crashes when statics depend on each other.
    // It can also be slow. In this case, we have no external dependencies so
    // it's safe. The cost is freeing our data (which is useless at shutdown),
    // but not terribly costly.
    // If we're deeply concerned, there are at least two ways to get around this.
    // (1) Explicitly instantiate and teardown statics from main().
    // (2) Observe that leaking this object at shutdown is really fine and just
    //     initialize it using a raw new.
    // The first one is a lot of work and the second will probably make static
    // analysis complain so I'm just going to tell the compiler to hush.
#pragma clang diagnostic ignored "-Wexit-time-destructors"
    static std::unique_ptr<ProgramIROperatorTypeConverter> instance;
#pragma clang diagnostic pop
    static std::once_flag flag;

    std::call_once(flag, []() {
        instance = std::unique_ptr<ProgramIROperatorTypeConverter>(new ProgramIROperatorTypeConverter());
    });

    return *instance;
}

IROperatorType ProgramIROperatorTypeConverter::GetType(const std::string& name) const
{
    auto opType = m_nameToType.find(name);
    if (opType == m_nameToType.cend()) {
        throw std::out_of_range("Unknown operator '" + name + "'");
    }

    return opType->second;
}

/*static*/ ProgramIROperatorTypeConverter::NameToTypeMap ProgramIROperatorTypeConverter::MakeNameToTypeMap()
{
    ProgramIROperatorTypeConverter::NameToTypeMap map{
        { "activation", IROperatorType::Activation },
        { "add", IROperatorType::Add },
        { "const", IROperatorType::Const },
        { "convolution", IROperatorType::Convolution },
        { "linear", IROperatorType::InnerProduct },
        { "matmul", IROperatorType::MatMul },
        { "pooling", IROperatorType::Pooling },
        { "softmax", IROperatorType::Softmax }
    };

    assert(map.size() == static_cast<size_t>(IROperatorType::COUNT));
    return map;
}
