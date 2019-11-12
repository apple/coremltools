//
//  IRValueType.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValueType.hpp"

using namespace ::CoreML::ILIL;

IRDimension::~IRDimension() = default;
IRDimension::IRDimension() = default;

//-----------------------------------------------------------------

IRConstantDimension::~IRConstantDimension() = default;

IRConstantDimension::IRConstantDimension(int64_t size)
    : m_size(size)
{ }

int64_t IRConstantDimension::GetSize() const
{
    return m_size;
}

bool IRConstantDimension::operator==(const IRDimension& other) const
{
    const auto* otherDim = dynamic_cast<const IRConstantDimension*>(&other);
    return (otherDim && GetSize() == otherDim->GetSize());
}

//-----------------------------------------------------------------

IRSymbolicDimension::~IRSymbolicDimension() = default;

IRSymbolicDimension::IRSymbolicDimension(const std::string& name)
    : m_name(name)
{ }

const std::string& IRSymbolicDimension::GetName() const
{
    return m_name;
}

bool IRSymbolicDimension::operator==(const IRDimension& other) const
{
    const auto* otherDim = dynamic_cast<const IRSymbolicDimension*>(&other);
    return (otherDim && GetName() == otherDim->GetName());
}

//-----------------------------------------------------------------

IRValueType::~IRValueType() = default;
IRValueType::IRValueType() = default;

bool IRValueType::operator!=(const IRValueType& other) const
{
    return !(*this == other);
}

//-----------------------------------------------------------------

IRScalarValueType::~IRScalarValueType() = default;

IRScalarValueType::IRScalarValueType(IRScalarValueTypeEnum type)
    : m_type(type)
{ }

IRScalarValueTypeEnum IRScalarValueType::GetType() const
{
    return m_type;
}

bool IRScalarValueType::operator==(const IRValueType& other) const
{
    const auto* otherScalarType = dynamic_cast<const IRScalarValueType*>(&other);
    return (otherScalarType && otherScalarType->m_type == m_type);
}

//-----------------------------------------------------------------

IRTensorValueType::~IRTensorValueType() = default;

IRTensorValueType::IRTensorValueType(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape)
    : m_scalarType(std::move(scalarType))
    , m_shape(std::move(shape))
{ }

const IRScalarValueType& IRTensorValueType::GetScalarType() const
{
    return *m_scalarType;
}

const IRTensorValueType::Shape& IRTensorValueType::GetShape() const
{
    return m_shape;
}

bool IRTensorValueType::operator==(const IRValueType& other) const
{
    if (const auto* otherTensorType = dynamic_cast<const IRTensorValueType*>(&other)) {
        return (GetScalarType() == otherTensorType->GetScalarType() &&
                GetShape() == otherTensorType->GetShape());
    }
    return false;
}

//-----------------------------------------------------------------

IRListValueType::~IRListValueType() = default;

IRListValueType::IRListValueType(std::shared_ptr<const IRValueType> elementType,
                                std::shared_ptr<const IRDimension> length)
    : m_elementType(std::move(elementType))
    , m_length(std::move(length))
{ }

const IRValueType& IRListValueType::GetElementType() const
{
    return *m_elementType;
}

const IRDimension& IRListValueType::GetLength() const
{
    return *m_length;
}

bool IRListValueType::operator==(const IRValueType& other) const
{
    if (const auto* otherListType = dynamic_cast<const IRListValueType*>(&other)) {
        return (GetElementType() == otherListType->GetElementType() &&
                GetLength() == otherListType->GetLength());
    }
    return false;
}

//-----------------------------------------------------------------

IRTupleValueType::~IRTupleValueType() = default;

IRTupleValueType::IRTupleValueType(ValueTypePtrVec&& types)
    : m_types(std::move(types))
{ }

const IRTupleValueType::ValueTypePtrVec& IRTupleValueType::GetTypes() const
{
    return m_types;
}

bool IRTupleValueType::operator==(const IRValueType& other) const
{
    if (const auto* otherTupleType = dynamic_cast<const IRTupleValueType*>(&other)) {
        return GetTypes() == otherTupleType->GetTypes();
    }
    return false;
}

//-----------------------------------------------------------------

IRNamedValueType::~IRNamedValueType() = default;

IRNamedValueType::IRNamedValueType(const std::string& name,
                                   std::shared_ptr<const IRValueType> type)
    : m_name(name)
    , m_type(std::move(type))
{ }

const std::string& IRNamedValueType::GetName() const
{
    return m_name;
}

const IRValueType& IRNamedValueType::GetType() const
{
    return *m_type;
}
