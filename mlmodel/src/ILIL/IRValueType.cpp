//
//  IRValueType.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValueType.hpp"

#include "ILIL/IRValue.hpp"
#include "ILIL/IRValueTypeTraits.hpp"
#include "NeuralNetworkBuffer.hpp"

#include <sys/stat.h>

using namespace ::CoreML::ILIL;

static void EnsureFileExists(const std::string& filePath)
{
    struct stat buffer;
    if (stat(filePath.c_str(), &buffer) != 0) {
        auto msg = "Could not open " + filePath;
        throw std::runtime_error(msg.c_str());
    }
}

IRDimension::~IRDimension() = default;
IRDimension::IRDimension() = default;

bool IRDimension::operator!=(const IRDimension& other) const
{
    return !(*this == other);
}

//-----------------------------------------------------------------

IRConstantDimension::~IRConstantDimension() = default;

IRConstantDimension::IRConstantDimension(uint64_t size)
    : m_size(size)
{ }

/*static*/ std::unique_ptr<IRConstantDimension> IRConstantDimension::Make(uint64_t size)
{
    return std::make_unique<IRConstantDimension>(size);
}

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

std::unique_ptr<IRSymbolicDimension> IRSymbolicDimension::Make(const std::string& name)
{
    return std::make_unique<IRSymbolicDimension>(name);
}

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

template<typename ScalarT>
std::unique_ptr<const IRScalarValue<ScalarT>> IRScalarValueType::Make(ScalarT value) const
{
    if (GetType() != CppTypeTraits<ScalarT>::IRTypeEnum) {
        throw std::runtime_error("Cannot initialize scalar value from value with wrong type.");
    }

    return std::unique_ptr<IRScalarValue<ScalarT>>
        (new IRScalarValue<ScalarT>(std::dynamic_pointer_cast<const IRScalarValueType>(shared_from_this()), value));
}

template std::unique_ptr<const IRScalarValue<float>> IRScalarValueType::Make(float value) const;
template std::unique_ptr<const IRScalarValue<double>> IRScalarValueType::Make(double value) const;
template std::unique_ptr<const IRScalarValue<int8_t>> IRScalarValueType::Make(int8_t value) const;
template std::unique_ptr<const IRScalarValue<int16_t>> IRScalarValueType::Make(int16_t value) const;
template std::unique_ptr<const IRScalarValue<int32_t>> IRScalarValueType::Make(int32_t value) const;
template std::unique_ptr<const IRScalarValue<int64_t>> IRScalarValueType::Make(int64_t value) const;
template std::unique_ptr<const IRScalarValue<uint8_t>> IRScalarValueType::Make(uint8_t value) const;
template std::unique_ptr<const IRScalarValue<uint16_t>> IRScalarValueType::Make(uint16_t value) const;
template std::unique_ptr<const IRScalarValue<uint32_t>> IRScalarValueType::Make(uint32_t value) const;
template std::unique_ptr<const IRScalarValue<uint64_t>> IRScalarValueType::Make(uint64_t value) const;
template std::unique_ptr<const IRScalarValue<bool>> IRScalarValueType::Make(bool value) const;
template std::unique_ptr<const IRScalarValue<std::string>> IRScalarValueType::Make(std::string value) const;

uint64_t IRScalarValueType::GetNumElements() const
{
    return 1;
}

IRScalarValueTypeEnum IRScalarValueType::GetType() const
{
    return m_type;
}

template<typename ScalarT>
static std::unique_ptr<const IRValue> ReadScalarValue(const std::string& filePath, uint64_t offset, const IRScalarValueType& type)
{
    EnsureFileExists(filePath);

    std::vector<ScalarT> value;
    NNBuffer::NeuralNetworkBuffer nnBuffer(filePath, /*readOnly=*/ true);
    nnBuffer.getBuffer(offset, value);
    auto retval = IRScalarValue<ScalarT>::Make(value.at(0));

    if (retval->GetType() != type) {
        throw std::logic_error("Wrong specialization of ReadScalarValue called for given IRScalarValueType.");
    }

    return retval;
}

std::unique_ptr<const IRValue> IRScalarValueType::ReadValue(const std::string& filePath, uint64_t offset) const
{
    switch (GetType()) {
        case IRScalarValueTypeEnum::Float32:
            return ReadScalarValue<float>(filePath, offset, *this);
        case IRScalarValueTypeEnum::Float64:
            return ReadScalarValue<double>(filePath, offset, *this);

        case IRScalarValueTypeEnum::Int32:
            return ReadScalarValue<int32_t>(filePath, offset, *this);
        case IRScalarValueTypeEnum::Int64:
            return ReadScalarValue<int64_t>(filePath, offset, *this);

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

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Dynamic()
{
    return std::unique_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Dynamic));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Bool()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Bool));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::String()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::String));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Float16()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Float16));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Float32()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Float32));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Float64()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Float64));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::BFloat16()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::BFloat16));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Int4()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Int4));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Int8()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Int8));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Int16()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Int16));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Int32()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Int32));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::Int64()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::Int64));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::UInt4()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::UInt4));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::UInt8()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::UInt8));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::UInt16()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::UInt16));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::UInt32()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::UInt32));
}

/*static*/ std::shared_ptr<const IRScalarValueType> IRScalarValueType::UInt64()
{
    return std::shared_ptr<IRScalarValueType>(new IRScalarValueType(IRScalarValueTypeEnum::UInt64));
}

//-----------------------------------------------------------------

IRTensorValueType::~IRTensorValueType() = default;

IRTensorValueType::IRTensorValueType(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape)
    : m_scalarType(std::move(scalarType))
    , m_shape(std::move(shape))
{ }

/*static*/ std::shared_ptr<const IRTensorValueType>
IRTensorValueType::Make(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape)
{
    return std::shared_ptr<const IRTensorValueType>(new IRTensorValueType(std::move(scalarType), std::move(shape)));
}

template<typename ScalarT>
std::unique_ptr<const IRTensorValue<ScalarT>> IRTensorValueType::Make(std::vector<ScalarT>&& values) const
{
    if (GetNumElements() != values.size()) {
        throw std::range_error("Wrong number of elements specified for immediate tensor value.");
    }

    if (GetScalarType().GetType() != CppTypeTraits<ScalarT>::IRTypeEnum) {
        throw std::runtime_error("Cannot initialize tensor value from values with wrong type.");
    }
    
    return std::unique_ptr<IRTensorValue<ScalarT>>
        (new IRTensorValue<ScalarT>(std::dynamic_pointer_cast<const IRTensorValueType>(shared_from_this()), std::move(values)));
}

template std::unique_ptr<const IRTensorValue<float>> IRTensorValueType::Make(std::vector<float>&& values) const;
template std::unique_ptr<const IRTensorValue<double>> IRTensorValueType::Make(std::vector<double>&& values) const;
template std::unique_ptr<const IRTensorValue<int32_t>> IRTensorValueType::Make(std::vector<int32_t>&& values) const;
template std::unique_ptr<const IRTensorValue<int64_t>> IRTensorValueType::Make(std::vector<int64_t>&& values) const;
template std::unique_ptr<const IRTensorValue<bool>> IRTensorValueType::Make(std::vector<bool>&& values) const;
template std::unique_ptr<const IRTensorValue<std::string>> IRTensorValueType::Make(std::vector<std::string>&& values) const;

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
static std::unique_ptr<const IRValue> ReadTensorValue(const std::string& filePath, uint64_t offset, const IRTensorValueType& tensorType)
{
    EnsureFileExists(filePath);

    std::vector<ScalarT> values;
    NNBuffer::NeuralNetworkBuffer nnBuffer(filePath, /*readOnly=*/ true);
    nnBuffer.getBuffer(offset, values);
    return tensorType.Make(std::move(values));
}

std::unique_ptr<const IRValue> IRTensorValueType::ReadValue(const std::string& filePath, uint64_t offset) const
{
    switch (GetScalarType().GetType()) {
        case IRScalarValueTypeEnum::Float32:
            return ReadTensorValue<float>(filePath, offset, *this->As<IRTensorValueType>());
        case IRScalarValueTypeEnum::Float64:
            return ReadTensorValue<double>(filePath, offset, *this->As<IRTensorValueType>());

        case IRScalarValueTypeEnum::Int32:
            return ReadTensorValue<int32_t>(filePath, offset, *this->As<IRTensorValueType>());
        case IRScalarValueTypeEnum::Int64:
            return ReadTensorValue<int64_t>(filePath, offset, *this->As<IRTensorValueType>());

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
        if (GetScalarType() != otherTensorType->GetScalarType()) {
            return false;
        }

        if (GetShape().size() != otherTensorType->GetShape().size()) {
            return false;
        }

        for (size_t i = 0; i < GetShape().size(); ++i) {
            if (*GetShape()[i] != *otherTensorType->GetShape()[i]) {
                return false;
            }
        }

        return true;
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

/** Convenience factory method. */
/*static*/ std::shared_ptr<const IRListValueType> IRListValueType::Make(std::shared_ptr<const IRValueType> elementType,
                                                                        std::shared_ptr<const IRDimension> length)
{
    return std::shared_ptr<IRListValueType>(new IRListValueType(std::move(elementType), std::move(length)));
}

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

/*static*/ std::shared_ptr<const IRTupleValueType> IRTupleValueType::Make(ValueTypePtrVec&& types)
{
    return std::shared_ptr<IRTupleValueType>(new IRTupleValueType(std::move(types)));
}

std::unique_ptr<const IRTupleValue> IRTupleValueType::Make(ConstIRValueVec&& values) const
{
    if (GetTypes().size() != values.size()) {
        throw std::runtime_error("Unexpected immediate tuple value type.");
    }

    for (size_t i = 0; i < GetTypes().size(); ++i) {
        if (*GetTypes()[i] != values[i]->GetType()) {
            throw std::runtime_error("Unexpected immediate tuple value type.");
        }
    }

    return std::unique_ptr<IRTupleValue>
        (new IRTupleValue(std::dynamic_pointer_cast<const IRTupleValueType>(shared_from_this()), std::move(values)));
}

uint64_t IRTupleValueType::GetNumElements() const
{
    throw std::range_error("Cannot determine number of elements in a tuple.");
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
        if (GetTypes().size() != otherTupleType->GetTypes().size()) {
            return false;
        }

        for (size_t i = 0; i < GetTypes().size(); ++i) {
            if (*GetTypes()[i] != *otherTupleType->GetTypes()[i]) {
                return false;
            }
        }
        return true;
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

/*static*/ std::shared_ptr<const IRNamedValueType>
IRNamedValueType::Make(const std::string& name, std::shared_ptr<const IRValueType> type)
{
    return std::unique_ptr<const IRNamedValueType>(new IRNamedValueType(name, std::move(type)));
}

const std::string& IRNamedValueType::GetName() const
{
    return m_name;
}

const IRValueType& IRNamedValueType::GetType() const
{
    return *m_type;
}
