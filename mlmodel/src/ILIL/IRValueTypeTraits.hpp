//
//  IRValueTypeTraits.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/16/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRValueType.hpp"

namespace CoreML {
namespace ILIL {

/**
 ILIL type info for native C++ types.
 */
template<typename CppT>
struct CppTypeTraits { };

template<>
struct CppTypeTraits<bool> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Bool;
};

template<>
struct CppTypeTraits<std::string> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::String;
};

template<>
struct CppTypeTraits<float> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Float32;
};

template<>
struct CppTypeTraits<double> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Float64;
};

template<>
struct CppTypeTraits<int8_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Int8;
};

template<>
struct CppTypeTraits<int16_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Int16;
};

template<>
struct CppTypeTraits<int32_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Int32;
};

template<>
struct CppTypeTraits<int64_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::Int64;
};

template<>
struct CppTypeTraits<uint8_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::UInt8;
};

template<>
struct CppTypeTraits<uint16_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::UInt16;
};

template<>
struct CppTypeTraits<uint32_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::UInt32;
};

template<>
struct CppTypeTraits<uint64_t> {
    constexpr static auto IRTypeEnum = IRScalarValueTypeEnum::UInt64;
};

}
}
