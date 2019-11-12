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
            return std::make_unique<IRConstantDimension>(specDim.size());
        case V5::Dimension::kSymbol:
            return std::make_unique<IRSymbolicDimension>(specDim.symbol());
        case V5::Dimension::DIMENSION_NOT_SET:
            throw std::runtime_error("Cannot parse invalid Dimension.");
    }
}

static std::unique_ptr<IRListValueType> ParseListType(const V5::ListType& specType)
{
    auto elementType = ProgramIRValueType::Parse(specType.type());
    auto length = ParseDimension(specType.length());
    return std::make_unique<IRListValueType>(std::move(elementType), std::move(length));
}

static std::unique_ptr<IRScalarValueType> ParseScalarType(V5::ScalarType specType)
{
    IRScalarValueTypeEnum scalarType;
    switch (specType) {
        case V5::DYNAMIC:
            scalarType = IRScalarValueTypeEnum::Dynamic;
            break;
        case V5::STRING:
            scalarType = IRScalarValueTypeEnum::String;
            break;
        case V5::BOOL:
            scalarType = IRScalarValueTypeEnum::Bool;
            break;
        case V5::FLOAT16:
            scalarType = IRScalarValueTypeEnum::Float16;
            break;
        case V5::FLOAT32:
            scalarType = IRScalarValueTypeEnum::Float32;
            break;
        case V5::FLOAT64:
            scalarType = IRScalarValueTypeEnum::Float64;
            break;
        case V5::BFLOAT16:
            scalarType = IRScalarValueTypeEnum::BFloat16;
            break;
        case V5::INT4:
            scalarType = IRScalarValueTypeEnum::Int4;
            break;
        case V5::INT8:
            scalarType = IRScalarValueTypeEnum::Int8;
            break;
        case V5::INT16:
            scalarType = IRScalarValueTypeEnum::Int16;
            break;
        case V5::INT32:
            scalarType = IRScalarValueTypeEnum::Int32;
            break;
        case V5::INT64:
            scalarType = IRScalarValueTypeEnum::Int64;
            break;
        case V5::UINT4:
            scalarType = IRScalarValueTypeEnum::UInt4;
            break;
        case V5::UINT8:
            scalarType = IRScalarValueTypeEnum::UInt8;
            break;
        case V5::UINT16:
            scalarType = IRScalarValueTypeEnum::UInt16;
            break;
        case V5::UINT32:
            scalarType = IRScalarValueTypeEnum::UInt32;
            break;
        case V5::UINT64:
            scalarType = IRScalarValueTypeEnum::UInt64;
            break;

        case V5::ScalarType_INT_MAX_SENTINEL_DO_NOT_USE_:
        case V5::ScalarType_INT_MIN_SENTINEL_DO_NOT_USE_:
            throw std::runtime_error("Found invalid scalar type.");
    }

    return std::make_unique<IRScalarValueType>(scalarType);
}

static std::unique_ptr<IRTensorValueType> ParseTensorType(const V5::TensorType& specType)
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

    return std::make_unique<IRTensorValueType>(std::move(scalarType), std::move(shape));
}

static std::unique_ptr<IRTupleValueType> ParseTupleType(const V5::TupleType& specType)
{
    IRTupleValueType::ValueTypePtrVec memberTypes;
    memberTypes.reserve(static_cast<size_t>(specType.values_size()));
    for (const auto& specMemberType : specType.values()) {
        memberTypes.push_back(ProgramIRValueType::Parse(specMemberType));
    }
    return std::make_unique<IRTupleValueType>(std::move(memberTypes));
}

std::unique_ptr<IRValueType> ProgramIRValueType::Parse(const SpecValueType& type)
{
    switch (type.type_case()) {
        case SpecValueType::kListType:
            return ParseListType(type.listtype());
        case SpecValueType::kScalarType:
            return ParseScalarType(type.scalartype());
        case SpecValueType::kTensorType:
            return ParseTensorType(type.tensortype());
        case SpecValueType::kTupleType:
            return ParseTupleType(type.tupletype());
        case SpecValueType::TYPE_NOT_SET:
            throw std::runtime_error("Cannot parse invalid value type");
    }
}
