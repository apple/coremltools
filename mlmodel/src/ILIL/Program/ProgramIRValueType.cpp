//
//  ProgramIRValueType.cpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/9/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/Program/ProgramIRValueType.hpp"

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

static std::unique_ptr<IRDimension> ParseDimension(const V5::Dimension& specDim)
{
    switch (specDim.dimension_case()) {
        case V5::Dimension::kSize:
            return std::make_unique<IRConstantDimension>(static_cast<uint64_t>(specDim.size()));
        case V5::Dimension::kSymbol:
            return std::make_unique<IRSymbolicDimension>(specDim.symbol());
        case V5::Dimension::DIMENSION_NOT_SET:
            throw std::runtime_error("Cannot parse invalid Dimension.");
    }
}

static std::shared_ptr<const IRListValueType> ParseListType(const V5::ListType& specType)
{
    auto elementType = ProgramIRValueType::Parse(specType.type());
    auto length = ParseDimension(specType.length());
    return IRListValueType::Make(std::move(elementType), std::move(length));
}

static IRScalarValueTypeEnum ParseScalarType(V5::ScalarType specType)
{
    switch (specType) {
        case V5::DYNAMIC:
            return IRScalarValueTypeEnum::Dynamic;
        case V5::STRING:
            return IRScalarValueTypeEnum::String;
        case V5::BOOL:
            return IRScalarValueTypeEnum::Bool;
        case V5::FLOAT16:
            return IRScalarValueTypeEnum::Float16;
        case V5::FLOAT32:
            return IRScalarValueTypeEnum::Float32;
        case V5::FLOAT64:
            return IRScalarValueTypeEnum::Float64;
        case V5::BFLOAT16:
            return IRScalarValueTypeEnum::BFloat16;
        case V5::INT4:
            return IRScalarValueTypeEnum::Int4;
        case V5::INT8:
            return IRScalarValueTypeEnum::Int8;
        case V5::INT16:
            return IRScalarValueTypeEnum::Int16;
        case V5::INT32:
            return IRScalarValueTypeEnum::Int32;
        case V5::INT64:
            return IRScalarValueTypeEnum::Int64;
        case V5::UINT4:
            return IRScalarValueTypeEnum::UInt4;
        case V5::UINT8:
            return IRScalarValueTypeEnum::UInt8;
        case V5::UINT16:
            return IRScalarValueTypeEnum::UInt16;
        case V5::UINT32:
            return IRScalarValueTypeEnum::UInt32;
        case V5::UINT64:
            return IRScalarValueTypeEnum::UInt64;

        case V5::ScalarType_INT_MAX_SENTINEL_DO_NOT_USE_:
        case V5::ScalarType_INT_MIN_SENTINEL_DO_NOT_USE_:
            throw std::runtime_error("Found invalid scalar type.");
    }
}

static std::shared_ptr<const IRTensorValueType> ParseTensorType(const V5::TensorType& specType)
{
    if (specType.rank() != static_cast<int64_t>(specType.dimension_size())) {
        throw std::runtime_error("Mismatch between specified rank and given dimensions.");
    }

    auto scalarType = ParseScalarType(specType.scalartype());

    IRTensorValueType::Shape shape;
    shape.reserve(static_cast<size_t>(specType.dimension_size()));
    for (const auto& dim : specType.dimension()) {
        shape.push_back(ParseDimension(dim));
    }

    return IRTensorValueType::Make(std::move(scalarType), std::move(shape));
}

static std::shared_ptr<const IRTupleValueType> ParseTupleType(const V5::TupleType& specType)
{
    IRTupleValueType::ValueTypePtrVec memberTypes;
    memberTypes.reserve(static_cast<size_t>(specType.values_size()));
    for (const auto& specMemberType : specType.values()) {
        memberTypes.push_back(ProgramIRValueType::Parse(specMemberType));
    }
    return IRTupleValueType::Make(std::move(memberTypes));
}

std::shared_ptr<const IRValueType> ProgramIRValueType::Parse(const SpecValueType& type)
{
    switch (type.type_case()) {
        case SpecValueType::kListType:
            return ParseListType(type.listtype());
        case SpecValueType::kScalarType:
            return std::make_shared<const IRScalarValueType>(ParseScalarType(type.scalartype()));
        case SpecValueType::kTensorType:
            return ParseTensorType(type.tensortype());
        case SpecValueType::kTupleType:
            return ParseTupleType(type.tupletype());
        case SpecValueType::TYPE_NOT_SET:
            throw std::runtime_error("Cannot parse invalid value type");
    }
}
