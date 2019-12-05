//
//  IRValue.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValue.hpp"
#include "ILIL/IRValueType.hpp"

using namespace ::CoreML::ILIL;

IRValue::~IRValue() = default;
IRValue::IRValue(ConstIRValueTypePtr type)
    : m_type(std::move(type))
{ }

const IRValueType& IRValue::GetType() const
{
    return *m_type;
}

const IRValue::ConstIRValueTypePtr& IRValue::GetTypePtr() const
{
    return m_type;
}

bool IRValue::AsBool() const
{
    throw std::bad_cast();
}

float IRValue::AsFloat32() const
{
    throw std::bad_cast();
}

int64_t IRValue::AsInt64() const
{
    throw std::bad_cast();
}

std::string IRValue::AsString() const
{
    throw std::bad_cast();
}

//-----------------------------------------------------------------
#pragma mark - IRScalarValue

template<typename T>
IRScalarValue<T>::~IRScalarValue() = default;

template<typename T>
IRScalarValue<T>::IRScalarValue(ConstIRScalarValueTypePtr type, T value)
    : IRValue(std::move(type))
    , m_value(value)
{ }

template<>
/*static*/ std::unique_ptr<IRScalarValue<float>>
IRScalarValue<float>::Make(float value)
{
    return std::unique_ptr<IRScalarValue<float>>
        (new IRScalarValue<float>(IRScalarValueType::Float32(), value));
}

template<>
/*static*/ std::unique_ptr<IRScalarValue<double>>
IRScalarValue<double>::Make(double value)
{
    return std::unique_ptr<IRScalarValue<double>>
        (new IRScalarValue<double>(IRScalarValueType::Float64(), value));
}

template<>
/*static*/ std::unique_ptr<IRScalarValue<int32_t>>
IRScalarValue<int32_t>::Make(int32_t value)
{
    return std::unique_ptr<IRScalarValue<int32_t>>
        (new IRScalarValue<int32_t>(IRScalarValueType::Int32(), value));
}

template<>
/*static*/ std::unique_ptr<IRScalarValue<int64_t>>
IRScalarValue<int64_t>::Make(int64_t value)
{
    return std::unique_ptr<IRScalarValue<int64_t>>
        (new IRScalarValue<int64_t>(IRScalarValueType::Int64(), value));
}

template<>
/*static*/ std::unique_ptr<IRScalarValue<bool>>
IRScalarValue<bool>::Make(bool value)
{
    return std::unique_ptr<IRScalarValue<bool>>
        (new IRScalarValue<bool>(IRScalarValueType::Bool(), value));
}

template<>
/*static*/ std::unique_ptr<IRScalarValue<std::string>>
IRScalarValue<std::string>::Make(std::string value)
{
    return std::unique_ptr<IRScalarValue<std::string>>
        (new IRScalarValue<std::string>(IRScalarValueType::String(), value));
}

// General-case implementations of scalar convenience getters: fail
template<typename T>
bool IRScalarValue<T>::AsBool() const
{
    return IRValue::AsBool();
}

template<typename T>
float IRScalarValue<T>::AsFloat32() const
{
    return IRValue::AsFloat32();
}

template<typename T>
int64_t IRScalarValue<T>::AsInt64() const
{
    return IRValue::AsInt64();
}

template<typename T>
std::string IRScalarValue<T>::AsString() const
{
    return IRValue::AsString();
}

// Specializations to actually fetch values
template<>
bool IRScalarValue<bool>::AsBool() const
{
    return GetValue();
}

template<>
float IRScalarValue<float>::AsFloat32() const
{
    return GetValue();
}

template<>
int64_t IRScalarValue<int64_t>::AsInt64() const
{
    return GetValue();
}

template<>
std::string IRScalarValue<std::string>::AsString() const
{
    return GetValue();
}

template<typename T>
void IRScalarValue<T>::CopyTo(void* dest, uint64_t destSize) const
{
    if (sizeof(T) > destSize) {
        throw std::runtime_error("Insufficient output buffer size.");
    }
    memcpy(dest, &m_value, sizeof(T));
}

template<typename T>
T IRScalarValue<T>::GetValue() const
{
    return m_value;
}

template class ::CoreML::ILIL::IRScalarValue<float>;
template class ::CoreML::ILIL::IRScalarValue<double>;
template class ::CoreML::ILIL::IRScalarValue<int8_t>;
template class ::CoreML::ILIL::IRScalarValue<int16_t>;
template class ::CoreML::ILIL::IRScalarValue<int32_t>;
template class ::CoreML::ILIL::IRScalarValue<int64_t>;
template class ::CoreML::ILIL::IRScalarValue<uint8_t>;
template class ::CoreML::ILIL::IRScalarValue<uint16_t>;
template class ::CoreML::ILIL::IRScalarValue<uint32_t>;
template class ::CoreML::ILIL::IRScalarValue<uint64_t>;
template class ::CoreML::ILIL::IRScalarValue<bool>;
template class ::CoreML::ILIL::IRScalarValue<std::string>;

//-----------------------------------------------------------------
#pragma mark - IRTensorValue

template<typename T>
IRTensorValue<T>::~IRTensorValue() = default;

template<typename T>
IRTensorValue<T>::IRTensorValue(ConstIRTensorValueTypePtr type, std::vector<T>&& values)
    : IRValue(type)
    , m_values(std::move(values))
{ }

template<typename T>
void IRTensorValue<T>::CopyTo(void* dest, uint64_t destSize) const
{
    uint64_t dataSize = GetType().GetNumElements() * sizeof(T);
    if (dataSize > destSize) {
        throw std::runtime_error("Insufficient output buffer size.");
    }
    memcpy(dest, m_values.data(), dataSize);
}

template<>
void IRTensorValue<bool>::CopyTo(void* /*dest*/, uint64_t /*destSize*/) const
{
    // std::vector<bool> is a bitfield so it doesn't have a data() member
    throw std::runtime_error("Copying boolean tensors is not supported.");
}

template<>
void IRTensorValue<std::string>::CopyTo(void* /*dest*/, uint64_t /*destSize*/) const
{
    // We have not yet designed a portable in-memory string format.
    throw std::runtime_error("Copying string tensors is not supported.");
}

template<typename T>
const std::vector<T>& IRTensorValue<T>::GetValues() const
{
    return m_values;
}

template class ::CoreML::ILIL::IRTensorValue<float>;
template class ::CoreML::ILIL::IRTensorValue<double>;
template class ::CoreML::ILIL::IRTensorValue<int32_t>;
template class ::CoreML::ILIL::IRTensorValue<int64_t>;
template class ::CoreML::ILIL::IRTensorValue<bool>;
template class ::CoreML::ILIL::IRTensorValue<std::string>;

//-----------------------------------------------------------------
#pragma mark - IRTupleValue

IRTupleValue::~IRTupleValue() = default;

IRTupleValue::IRTupleValue(ConstIRTupleValueTypePtr type, ConstIRValueVec&& values)
    : IRValue(std::move(type))
    , m_values(std::move(values))
{ }

void IRTupleValue::CopyTo(void* /*dest*/, uint64_t /*destSize*/) const
{
    throw std::runtime_error("Copying tuples is not supported.");
}

const IRTupleValue::ConstIRValueVec& IRTupleValue::GetValues() const
{
    return m_values;
}

//-----------------------------------------------------------------
#pragma mark - IRFileValue

IRFileValue::~IRFileValue() = default;

IRFileValue::IRFileValue(ConstIRValueTypePtr type, const std::string& path, uint64_t offset)
    : IRValue(std::move(type))
    , m_path(path)
    , m_offset(offset)
{ }

void IRFileValue::CopyTo(void* dest, uint64_t destSize) const
{
    auto value = GetType().ReadValue(GetPath(), GetOffset());
    value->CopyTo(dest, destSize);
}

const std::string& IRFileValue::GetPath() const
{
    return m_path;
}

uint64_t IRFileValue::GetOffset() const
{
    return m_offset;
}
