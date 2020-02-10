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

const IRScope* IRScope::GetParent() const
{
    return m_parent.get();
}

const IRValueType& IRScope::GetType(const std::string& name, bool includeRoot) const
{
    auto type = TryGetType(name, includeRoot);
    if (type) {
        return *type;
    }

    throw std::runtime_error("Failed to find type of " + name + '.');
}

IRScope::ConstIRValueTypePtr IRScope::GetTypeSharedPtr(const std::string& name, bool includeRoot) const
{
    auto type = TryGetTypeSharedPtr(name, includeRoot);
    if (type) {
        return type;
    }

    throw std::runtime_error("Failed to find type of " + name + '.');
}

const IRValueType* IRScope::TryGetType(const std::string& name, bool includeRoot) const
{
    return TryGetTypeSharedPtr(name, includeRoot).get();
}

IRScope::ConstIRValueTypePtr IRScope::TryGetTypeSharedPtr(const std::string& name, bool includeRoot) const
{
    auto localType = m_types.find(name);
    if (localType != m_types.cend()) {
        return (includeRoot || m_parent) ? localType->second : nullptr;
    }

    return m_parent ? m_parent->TryGetTypeSharedPtr(name, includeRoot) : nullptr;
}

const IRScope::TypeMap& IRScope::GetTypes() const
{
    return m_types;
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

const IRValue& IRScope::GetValue(const std::string& name, bool includeRoot) const
{
    auto value = TryGetValue(name, includeRoot);
    if (value) {
        return *value;
    }

    throw std::runtime_error("Failed to find value of " + name + '.');
}

const IRValue* IRScope::TryGetValue(const std::string& name, bool includeRoot) const
{
    return TryGetValueSharedPtr(name, includeRoot).get();
}

IRScope::ConstIRValuePtr IRScope::TryGetValueSharedPtr(const std::string& name, bool includeRoot) const
{
    auto localValue = m_values.find(name);
    if (localValue != m_values.cend()) {
        return (includeRoot || m_parent) ? localValue->second : nullptr;
    }

    return m_parent ? m_parent->TryGetValueSharedPtr(name, includeRoot) : nullptr;
}

const IRScope::ValueMap& IRScope::GetValues() const
{
    return m_values;
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

std::unique_ptr<IRScope> IRScope::WithRenames(const RenameVec& renames) const
{
    auto copy = std::make_unique<IRScope>(*this);

    for (const auto& oldAndNew : renames) {
        auto iter = copy->m_types.find(oldAndNew.first);
        if (iter != copy->m_types.end()) {
            copy->m_types[oldAndNew.second] = iter->second;
            copy->m_types.erase(oldAndNew.first);
        }
    }

    for (const auto& oldAndNew : renames) {
        auto iter = copy->m_values.find(oldAndNew.first);
        if (iter != copy->m_values.end()) {
            copy->m_values[oldAndNew.second] = iter->second;
            copy->m_values.erase(oldAndNew.first);
        }
    }

    return copy;
}
