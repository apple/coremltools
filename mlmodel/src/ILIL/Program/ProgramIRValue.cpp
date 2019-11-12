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

static std::unique_ptr<IRValue> ParseFileValue(const V5::Value_FileValue& value, std::unique_ptr<IRValueType> type)
{
    return std::make_unique<IRFileValue>(std::move(type), value.filename(), value.offset());
}

template<typename ScalarT>
static std::unique_ptr<IRImmediateValue> ParseScalarValue(std::shared_ptr<IRValueType> type, ScalarT value)
{
    auto scalarType = std::dynamic_pointer_cast<IRScalarValueType>(type);
    if (!scalarType) {
        throw std::runtime_error("Cannot parse scalar value with associated non-scalar type.");
    }

    return std::make_unique<IRImmediateScalarValue<ScalarT>>(std::move(scalarType), value);
}

static std::unique_ptr<IRImmediateValue> ParseTensorValue(const V5::TensorValue& value, std::shared_ptr<IRValueType> type)
{
    auto tensorType = std::dynamic_pointer_cast<const IRTensorValueType>(type);
    if (!tensorType) {
        throw std::runtime_error("Cannot parse tensor value with associated non-tensor type.");
    }

    switch (tensorType->GetScalarType().GetType()) {
        case IRScalarValueTypeEnum::Dynamic:
            throw std::runtime_error("Dynamic is not a supported immediate type.");

        case IRScalarValueTypeEnum::Bool:
        {
            std::vector<bool> bools(value.bools().cbegin(), value.bools().cend());
            return std::make_unique<IRImmediateTensorValue<bool>>(std::move(tensorType), std::move(bools));
        }

        case IRScalarValueTypeEnum::String:
        {
            std::vector<std::string> strings(value.strings().cbegin(), value.strings().cend());
            return std::make_unique<IRImmediateTensorValue<std::string>>(std::move(tensorType), std::move(strings));
        }

        case IRScalarValueTypeEnum::Float16:
            throw std::runtime_error("Float16 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Float32:
        {
            std::vector<float> floats(value.floats().cbegin(), value.floats().cend());
            return std::make_unique<IRImmediateTensorValue<float>>(std::move(tensorType), std::move(floats));
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
            throw std::runtime_error("Int32 is not a supported immediate type.");
        case IRScalarValueTypeEnum::Int64:
        {
            std::vector<int64_t> ints(value.ints().cbegin(), value.ints().cend());
            return std::make_unique<IRImmediateTensorValue<int64_t>>(std::move(tensorType), std::move(ints));
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
            throw std::runtime_error("UInt32 is not a supported immediate type.");
    }
}

static std::unique_ptr<IRImmediateValue> ParseTupleValue(const V5::TupleValue& value, std::shared_ptr<IRValueType> type)
{
    auto tupleType = std::dynamic_pointer_cast<const IRTupleValueType>(type);
    if (!tupleType) {
        throw std::runtime_error("Cannot parse tuple value with associated non-tuple type.");
    }

    const auto& memberTypes = tupleType->GetTypes();
    IRImmediateTupleValue::ConstIRValueVec values;
    values.reserve(memberTypes.size());
    for (int i = 0; i < value.value_size(); ++i) {
        values.push_back(ProgramIRValue::Parse(value.value(i)));

        if (values[static_cast<size_t>(i)]->GetType() != *memberTypes[static_cast<size_t>(i)]) {
            throw std::runtime_error("Found unexpected immediate type while initializing tuple.");
        }
    }

    return std::make_unique<IRImmediateTupleValue>(std::move(tupleType), std::move(values));
}

static std::unique_ptr<IRImmediateValue> ParseImmediateValue(const V5::Value_ImmediateValue& value, std::unique_ptr<IRValueType> type)
{
    switch (value.value_case()) {
        case V5::Value_ImmediateValue::kB:
            return ParseScalarValue<bool>(std::move(type), value.b());
        case V5::Value_ImmediateValue::kF:
            return ParseScalarValue<float>(std::move(type), value.f());
        case V5::Value_ImmediateValue::kI:
            return ParseScalarValue<int64_t>(std::move(type), value.i());
        case V5::Value_ImmediateValue::kS:
            return ParseScalarValue<std::string>(std::move(type), value.s());
        case V5::Value_ImmediateValue::kTensor:
            return ParseTensorValue(value.tensor(), std::move(type));
        case V5::Value_ImmediateValue::kTuple:
            return ParseTupleValue(value.tuple(), std::move(type));
        case V5::Value_ImmediateValue::VALUE_NOT_SET:
            throw std::runtime_error("Cannot parse invalid immediate value.");
    }
}

std::unique_ptr<IRValue> ProgramIRValue::Parse(const SpecValue& value)
{
    auto type = ProgramIRValueType::Parse(value.type());
    switch (value.value_case()) {
        case SpecValue::kFileValue:
            return ParseFileValue(value.filevalue(), std::move(type));
        case SpecValue::kImmediateValue:
            return ParseImmediateValue(value.immediatevalue(), std::move(type));
        case SpecValue::VALUE_NOT_SET:
            throw std::runtime_error("Cannot parse invalid value");
    }
}
