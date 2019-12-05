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

IRScope::ConstIRValueTypePtr IRScope::GetType(const std::string& name, bool includeRoot) const
{
    auto type = TryGetType(name, includeRoot);
    if (type) {
        return type;
    }

    throw std::runtime_error("Failed to find type of " + name + '.');
}

IRScope::ConstIRValueTypePtr IRScope::TryGetType(const std::string& name, bool includeRoot) const
{
    auto localType = m_types.find(name);
    if (localType != m_types.cend()) {
        return (includeRoot || m_parent) ? localType->second : nullptr;
    }

    return m_parent ? m_parent->TryGetType(name, includeRoot) : nullptr;
}

bool IRScope::SetType(const std::string& name, ConstIRValueTypePtr type, bool allowReplace)
{
    auto existsBeforeInsert = m_types.count(name) != 0;

    if (existsBeforeInsert && !allowReplace) {
        throw std::runtime_error("Type for '" + name + "' already found in scope.");
    }

    m_types[name] = std::move(type);
    return !existsBeforeInsert;
}

IRScope::ConstIRValuePtr IRScope::GetValue(const std::string& name, bool includeRoot) const
{
    auto value = TryGetValue(name, includeRoot);
    if (value) {
        return value;
    }

    throw std::runtime_error("Failed to find value of " + name + '.');
}

IRScope::ConstIRValuePtr IRScope::TryGetValue(const std::string& name, bool includeRoot) const
{
    auto localValue = m_values.find(name);
    if (localValue != m_values.cend()) {
        return (includeRoot || m_parent) ? localValue->second : nullptr;
    }

    return m_parent ? m_parent->TryGetValue(name, includeRoot) : nullptr;
}

bool IRScope::SetValue(const std::string& name, ConstIRValuePtr value, bool allowReplace)
{
    auto existsBeforeInsert = m_values.count(name) != 0;

    if (existsBeforeInsert && !allowReplace) {
        throw std::runtime_error("Value for '" + name + "' already found in scope.");
    }

    m_values[name] = std::move(value);
    return !existsBeforeInsert;
}
