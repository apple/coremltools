//
//  IRValueType.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValueType.hpp"

#include "ILIL/IRValue.hpp"
#include "NeuralNetworkBuffer.hpp"

using namespace ::CoreML::ILIL;

IRDimension::~IRDimension() = default;
IRDimension::IRDimension() = default;

//-----------------------------------------------------------------

IRConstantDimension::~IRConstantDimension() = default;

IRConstantDimension::IRConstantDimension(uint64_t size)
    : m_size(size)
{ }

uint64_t IRConstantDimension::GetSize() const
{
    return m_size;
}

bool IRConstantDimension::operator==(const IRDimension& other) const
{
    const auto* otherDim = other.TryAs<IRConstantDimension>();
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
    const auto* otherDim = other.TryAs<IRSymbolicDimension>();
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

uint64_t IRScalarValueType::GetNumElements() const
{
    return 1;
}

IRScalarValueTypeEnum IRScalarValueType::GetType() const
{
    return m_type;
}

template<typename ScalarT>
static std::unique_ptr<const IRValue> ReadScalarValue(const std::string& filePath, uint64_t offset, std::shared_ptr<const IRScalarValueType> type)
{
    std::vector<ScalarT> value;
    NNBuffer::NeuralNetworkBuffer nnBuffer(filePath);
    nnBuffer.getBuffer(offset, value);
    return std::make_unique<IRImmediateScalarValue<ScalarT>>(type, value.at(0));
}

std::unique_ptr<const IRValue> IRScalarValueType::ReadValue(const std::string& filePath, uint64_t offset) const
{
    auto thisTypeAsSharedPtr = std::make_shared<IRScalarValueType>(*this);
    switch (GetType()) {
        case IRScalarValueTypeEnum::Float32:
            return ReadScalarValue<float>(filePath, offset, thisTypeAsSharedPtr);
        case IRScalarValueTypeEnum::Float64:
            return ReadScalarValue<double>(filePath, offset, thisTypeAsSharedPtr);

        case IRScalarValueTypeEnum::Int32:
            return ReadScalarValue<int32_t>(filePath, offset, thisTypeAsSharedPtr);
        case IRScalarValueTypeEnum::Int64:
            return ReadScalarValue<int64_t>(filePath, offset, thisTypeAsSharedPtr);

        case IRScalarValueTypeEnum::Dynamic:
        case IRScalarValueTypeEnum::Bool:
        case IRScalarValueTypeEnum::String:
        case IRScalarValueTypeEnum::Float16:
        case IRScalarValueTypeEnum::BFloat16:
        case IRScalarValueTypeEnum::Int4:
        case IRScalarValueTypeEnum::Int8:
        case IRScalarValueTypeEnum::Int16:
        case IRScalarValueTypeEnum::UInt4:
        case IRScalarValueTypeEnum::UInt8:
        case IRScalarValueTypeEnum::UInt16:
        case IRScalarValueTypeEnum::UInt32:
        case IRScalarValueTypeEnum::UInt64:
            throw std::runtime_error("Reading binary data of the given type is not supported.");
    }
}

bool IRScalarValueType::operator==(const IRValueType& other) const
{
    const auto* otherScalarType = other.TryAs<IRScalarValueType>();
    return (otherScalarType && otherScalarType->m_type == m_type);
}

//-----------------------------------------------------------------

IRTensorValueType::~IRTensorValueType() = default;

IRTensorValueType::IRTensorValueType(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape)
    : m_scalarType(std::move(scalarType))
    , m_shape(std::move(shape))
{ }

uint64_t IRTensorValueType::GetNumElements() const
{
    uint64_t numElements = 1;
    for (const auto& dim : m_shape) {
        auto dimLength = dim->TryAs<IRConstantDimension>();
        if (dimLength) {
            numElements *= dimLength->GetSize();
        }
        else {
            throw std::range_error("Cannot determine number of elements in tensor with symbolic dimension.");
        }
    }
    return numElements;
}

const IRScalarValueType& IRTensorValueType::GetScalarType() const
{
    return *m_scalarType;
}

const IRTensorValueType::Shape& IRTensorValueType::GetShape() const
{
    return m_shape;
}

template<typename ScalarT>
static std::unique_ptr<const IRValue> ReadTensorValue(const std::string& filePath, uint64_t offset, std::shared_ptr<const IRTensorValueType> type)
{
    std::vector<ScalarT> values;
    NNBuffer::NeuralNetworkBuffer nnBuffer(filePath);
    nnBuffer.getBuffer(offset, values);
    return std::make_unique<IRImmediateTensorValue<ScalarT>>(type, std::move(values));
}

std::unique_ptr<const IRValue> IRTensorValueType::ReadValue(const std::string& filePath, uint64_t offset) const
{
    auto thisTypeAsSharedPtr = std::make_shared<IRTensorValueType>(*this);
    switch (GetScalarType().GetType()) {
        case IRScalarValueTypeEnum::Float32:
            return ReadTensorValue<float>(filePath, offset, thisTypeAsSharedPtr);
        case IRScalarValueTypeEnum::Float64:
            return ReadTensorValue<double>(filePath, offset, thisTypeAsSharedPtr);

        case IRScalarValueTypeEnum::Int32:
            return ReadTensorValue<int32_t>(filePath, offset, thisTypeAsSharedPtr);
        case IRScalarValueTypeEnum::Int64:
            return ReadTensorValue<int64_t>(filePath, offset, thisTypeAsSharedPtr);

        case IRScalarValueTypeEnum::Dynamic:
        case IRScalarValueTypeEnum::Bool:
        case IRScalarValueTypeEnum::String:
        case IRScalarValueTypeEnum::Float16:
        case IRScalarValueTypeEnum::BFloat16:
        case IRScalarValueTypeEnum::Int4:
        case IRScalarValueTypeEnum::Int8:
        case IRScalarValueTypeEnum::Int16:
        case IRScalarValueTypeEnum::UInt4:
        case IRScalarValueTypeEnum::UInt8:
        case IRScalarValueTypeEnum::UInt16:
        case IRScalarValueTypeEnum::UInt32:
        case IRScalarValueTypeEnum::UInt64:
            throw std::runtime_error("Reading binary data of the given type is not supported.");
    }
}

bool IRTensorValueType::operator==(const IRValueType& other) const
{
    if (const auto* otherTensorType = other.TryAs<IRTensorValueType>()) {
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

uint64_t IRListValueType::GetNumElements() const
{
    auto length = m_length->TryAs<IRConstantDimension>();
    if (length) {
        return length->GetSize();
    }

    throw std::range_error("Cannot determine number of elements in list with symbolic length.");
}

std::unique_ptr<const IRValue> IRListValueType::ReadValue(const std::string& /*filePath*/, uint64_t /*offset*/) const
{
    throw std::runtime_error("Reading binary list data is not supported.");
}

bool IRListValueType::operator==(const IRValueType& other) const
{
    if (const auto* otherListType = other.TryAs<IRListValueType>()) {
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

uint64_t IRTupleValueType::GetNumElements() const
{
    return static_cast<uint64_t>(m_types.size());
}

const IRTupleValueType::ValueTypePtrVec& IRTupleValueType::GetTypes() const
{
    return m_types;
}

std::unique_ptr<const IRValue> IRTupleValueType::ReadValue(const std::string& /*filePath*/, uint64_t /*offset*/) const
{
    throw std::runtime_error("Reading binary tuple data is not supported.");
}

bool IRTupleValueType::operator==(const IRValueType& other) const
{
    if (const auto* otherTupleType = other.TryAs<IRTupleValueType>()) {
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
