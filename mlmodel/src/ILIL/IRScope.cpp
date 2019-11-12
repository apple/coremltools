
//
//  IRScope.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRScope.hpp"

using namespace ::CoreML::ILIL;

IRScope::~IRScope() = default;

IRScope::IRScope(std::shared_ptr<const IRScope> parent)
    : m_parent(std::move(parent))
{ }

IRScope::ConstIRValueTypePtr IRScope::GetType(const std::string& name) const
{
    auto localType = m_types.find(name);
    if (localType != m_types.cend()) {
        return localType->second;
    }

    return m_parent ? m_parent->GetType(name) : nullptr;
}

void IRScope::SetType(const std::string& name, ConstIRValueTypePtr type)
{
    m_types[name] = std::move(type);
}

IRScope::ConstIRValuePtr IRScope::GetValue(const std::string& name) const
{
    auto localValue = m_values.find(name);
    if (localValue != m_values.cend()) {
        return localValue->second;
    }

    return m_parent ? m_parent->GetValue(name) : nullptr;
}

void IRScope::SetValue(const std::string& name, ConstIRValuePtr value)
{
    m_values[name] = std::move(value);
}
