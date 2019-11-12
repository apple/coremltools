//
//  IRValue.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValue.hpp"

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

IRImmediateValue::~IRImmediateValue() = default;

IRImmediateValue::IRImmediateValue(ConstIRValueTypePtr type)
    : IRValue(std::move(type))
{ }

//-----------------------------------------------------------------

template<typename T>
IRImmediateScalarValue<T>::~IRImmediateScalarValue() = default;

template<typename T>
IRImmediateScalarValue<T>::IRImmediateScalarValue(ConstIRScalarValueTypePtr type,
                                                  T value)
    : IRImmediateValue(std::move(type))
    , m_value(value)
{ }

// General-case implementations of scalar convenience getters: fail
template<typename T>
bool IRImmediateScalarValue<T>::AsBool() const
{
    return IRImmediateValue::AsBool();
}

template<typename T>
float IRImmediateScalarValue<T>::AsFloat32() const
{
    return IRImmediateValue::AsFloat32();
}

template<typename T>
int64_t IRImmediateScalarValue<T>::AsInt64() const
{
    return IRImmediateValue::AsInt64();
}

template<typename T>
std::string IRImmediateScalarValue<T>::AsString() const
{
    return IRImmediateValue::AsString();
}

// Specializations to actually fetch values
template<>
bool IRImmediateScalarValue<bool>::AsBool() const
{
    return GetValue();
}

template<>
float IRImmediateScalarValue<float>::AsFloat32() const
{
    return GetValue();
}

template<>
int64_t IRImmediateScalarValue<int64_t>::AsInt64() const
{
    return GetValue();
}

template<>
std::string IRImmediateScalarValue<std::string>::AsString() const
{
    return GetValue();
}

template<typename T>
T IRImmediateScalarValue<T>::GetValue() const
{
    return m_value;
}

template class ::CoreML::ILIL::IRImmediateScalarValue<float>;
template class ::CoreML::ILIL::IRImmediateScalarValue<int64_t>;
template class ::CoreML::ILIL::IRImmediateScalarValue<bool>;
template class ::CoreML::ILIL::IRImmediateScalarValue<std::string>;

//-----------------------------------------------------------------

template<typename T>
IRImmediateTensorValue<T>::~IRImmediateTensorValue() = default;

template<typename T>
IRImmediateTensorValue<T>::IRImmediateTensorValue(ConstIRTensorValueTypePtr type,
                                                  std::vector<T>&& values)
    : IRImmediateValue(type)
    , m_values(std::move(values))
{ }

template<typename T>
const std::vector<T>& IRImmediateTensorValue<T>::GetValues() const
{
    return m_values;
}

template class ::CoreML::ILIL::IRImmediateTensorValue<float>;
template class ::CoreML::ILIL::IRImmediateTensorValue<int64_t>;
template class ::CoreML::ILIL::IRImmediateTensorValue<bool>;
template class ::CoreML::ILIL::IRImmediateTensorValue<std::string>;

//-----------------------------------------------------------------

IRImmediateTupleValue::~IRImmediateTupleValue() = default;

IRImmediateTupleValue::IRImmediateTupleValue(ConstIRTupleValueTypePtr type,
                                             ConstIRValueVec&& values)
    : IRImmediateValue(std::move(type))
    , m_values(std::move(values))
{ }

//-----------------------------------------------------------------

IRFileValue::~IRFileValue() = default;

IRFileValue::IRFileValue(ConstIRValueTypePtr type, const std::string& path, uint64_t offset)
    : IRValue(std::move(type))
    , m_path(path)
    , m_offset(offset)
{ }

const std::string& IRFileValue::GetPath() const
{
    return m_path;
}

uint64_t IRFileValue::GetOffset() const
{
    return m_offset;
}
