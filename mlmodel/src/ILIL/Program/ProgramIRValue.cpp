//
//  ProgramIRValue.cpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIRValue.hpp"
#include "ILIL/Program/ProgramIRValueType.hpp"
#include "NeuralNetworkBuffer.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

static std::shared_ptr<const IRValue> ParseFileValue(const V5::Value_FileValue& value, std::shared_ptr<const IRValueType> type)
{
    return std::make_unique<IRFileValue>(std::move(type), value.filename(), value.offset());
}

static std::unique_ptr<const IRValue> ParseScalarValue(const IRScalarValueType& scalarType, bool value)
{
    if (scalarType.GetType() != IRScalarValueTypeEnum::Bool) {
        throw std::runtime_error("Cannot create non-boolean scalar value from boolean constant.");
    }
    return scalarType.Make(value);
}

static std::unique_ptr<const IRValue> ParseScalarValue(const IRScalarValueType& scalarType, float value)
{
    switch (scalarType.GetType()) {
        case IRScalarValueTypeEnum::Float32:
            return scalarType.Make(static_cast<float>(value));
        case IRScalarValueTypeEnum::Float64:
            return scalarType.Make(static_cast<double>(value));

        case IRScalarValueTypeEnum::Float16:
        case IRScalarValueTypeEnum::BFloat16:
            throw std::runtime_error("Cannot create scalar value for 16-bit float.");

        default:
            throw std::runtime_error("Cannot create non-float scalar value from float.");
    }
}

static std::unique_ptr<const IRValue> ParseScalarValue(const IRScalarValueType& scalarType, int64_t value)
{
    switch (scalarType.GetType()) {
        case IRScalarValueTypeEnum::Int8:
            return scalarType.Make(static_cast<int8_t>(value));
        case IRScalarValueTypeEnum::Int16:
            return scalarType.Make(static_cast<int16_t>(value));
        case IRScalarValueTypeEnum::Int32:
            return scalarType.Make(static_cast<int32_t>(value));
        case IRScalarValueTypeEnum::Int64:
            return scalarType.Make(static_cast<int64_t>(value));

        case IRScalarValueTypeEnum::UInt8:
            return scalarType.Make(static_cast<uint8_t>(value));
        case IRScalarValueTypeEnum::UInt16:
            return scalarType.Make(static_cast<uint16_t>(value));
        case IRScalarValueTypeEnum::UInt32:
            return scalarType.Make(static_cast<uint32_t>(value));
        case IRScalarValueTypeEnum::UInt64:
            return scalarType.Make(static_cast<uint64_t>(value));

        case IRScalarValueTypeEnum::Int4:
        case IRScalarValueTypeEnum::UInt4:
            throw std::runtime_error("Cannot create scalar value for 4-bit integer.");

        default:
            throw std::runtime_error("Cannot create non-integer scalar value from integer.");
    }
}

static std::unique_ptr<const IRValue> ParseScalarValue(const IRScalarValueType& scalarType, const std::string& value)
{
    if (scalarType.GetType() != IRScalarValueTypeEnum::String) {
        throw std::runtime_error("Cannot create non-string scalar value from string constant.");
    }
    return scalarType.Make(value);
}

static std::unique_ptr<const IRValue> ParseTensorValue(const V5::TensorValue& value, const IRTensorValueType& tensorType)
{
    switch (tensorType.GetScalarType()) {
        case IRScalarValueTypeEnum::Dynamic:
            throw std::runtime_error("Dynamic is not a supported immediate type.");

        case IRScalarValueTypeEnum::Bool:
        {
            std::vector<bool> bools(value.bools().cbegin(), value.bools().cend());
            return tensorType.MakeValue(std::move(bools));
        }

        case IRScalarValueTypeEnum::String:
        {
            std::vector<std::string> strings(value.strings().cbegin(), value.strings().cend());
            return tensorType.MakeValue(std::move(strings));
        }

        case IRScalarValueTypeEnum::Float16:
            throw std::runtime_error("Float16 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Float32:
        {
            std::vector<float> floats(value.floats().cbegin(), value.floats().cend());
            return tensorType.MakeValue(std::move(floats));
        }
        case IRScalarValueTypeEnum::Float64:
            throw std::runtime_error("Float64 is not a supported immediate type.");
        case IRScalarValueTypeEnum::BFloat16:
            throw std::runtime_error("BFloat16 is not a supported immediate type.");

        case IRScalarValueTypeEnum::Int4:
            throw std::runtime_error("Int4 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Int8:
            throw std::runtime_error("Int8 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Int16:
            throw std::runtime_error("Int16 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Int32:
        {
            std::vector<int32_t> ints(value.ints().cbegin(), value.ints().cend());
            return tensorType.MakeValue(std::move(ints));
        }
        case IRScalarValueTypeEnum::Int64:
        {
            std::vector<int64_t> ints(value.ints().cbegin(), value.ints().cend());
            return tensorType.MakeValue(std::move(ints));
        }

        case IRScalarValueTypeEnum::UInt4:
            throw std::runtime_error("UInt4 is not a supported immediate type.");
        case IRScalarValueTypeEnum::UInt8:
            throw std::runtime_error("UInt8 is not a supported immediate type.");
        case IRScalarValueTypeEnum::UInt16:
            throw std::runtime_error("UInt16 is not a supported immediate type.");
        case IRScalarValueTypeEnum::UInt32:
            throw std::runtime_error("UInt32 is not a supported immediate type.");
        case IRScalarValueTypeEnum::UInt64:
            throw std::runtime_error("UInt64 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Any:
            throw std::runtime_error("Any is not a supported immediate type.");
    }
}

static std::shared_ptr<const IRValue> ParseTupleValue(const V5::TupleValue& value, const IRTupleValueType& tupleType)
{
    const auto& memberTypes = tupleType.GetTypes();
    IRTupleValue::ConstIRValueVec values;
    values.reserve(memberTypes.size());
    for (int i = 0; i < value.value_size(); ++i) {
        values.push_back(ProgramIRValue::Parse(value.value(i)));
    }

    return tupleType.Make(std::move(values));
}

static std::shared_ptr<const IRValue> ParseImmediateValue(const V5::Value_ImmediateValue& value, const IRValueType& type)
{
    switch (value.value_case()) {
        case V5::Value_ImmediateValue::kB:
            return ParseScalarValue(*type.As<IRScalarValueType>(), value.b());
        case V5::Value_ImmediateValue::kF:
            return ParseScalarValue(*type.As<IRScalarValueType>(), value.f());
        case V5::Value_ImmediateValue::kI:
            return ParseScalarValue(*type.As<IRScalarValueType>(), value.i());
        case V5::Value_ImmediateValue::kS:
            return ParseScalarValue(*type.As<IRScalarValueType>(), value.s());
        case V5::Value_ImmediateValue::kTensor:
            return ParseTensorValue(value.tensor(), *type.As<IRTensorValueType>());
        case V5::Value_ImmediateValue::kTuple:
            return ParseTupleValue(value.tuple(), *type.As<IRTupleValueType>());
        case V5::Value_ImmediateValue::VALUE_NOT_SET:
            throw std::runtime_error("Cannot parse invalid immediate value.");
    }
}

std::shared_ptr<const IRValue> ProgramIRValue::Parse(const SpecValue& value)
{
    auto type = ProgramIRValueType::Parse(value.type());
    switch (value.value_case()) {
        case SpecValue::kFileValue:
            return ParseFileValue(value.filevalue(), std::move(type));
        case SpecValue::kImmediateValue:
            return ParseImmediateValue(value.immediatevalue(), *type);
        case SpecValue::VALUE_NOT_SET:
            throw std::runtime_error("Cannot parse invalid value");
    }
}
