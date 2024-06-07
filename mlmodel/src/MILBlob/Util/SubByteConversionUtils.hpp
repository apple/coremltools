// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Util/Span.hpp"
#include <vector>

namespace MILBlob {

// This header contains the utils used by coremltools to pack subbyte datatype values.

// Packs a span of int8_t containing unpacked values into a packed uint8_t vector
template <typename T>
std::vector<uint8_t> PackInt8Span(Util::Span<const int8_t> unpackedValues);

template <>
std::vector<uint8_t> PackInt8Span<Int4>(Util::Span<const int8_t> unpackedValues);

// Packs a span of uint8_t containing unpacked values into a packed uint8_t vector
template <typename T>
std::vector<uint8_t> PackUInt8Span(Util::Span<const uint8_t> unpackedValues);

template <>
std::vector<uint8_t> PackUInt8Span<UInt6>(Util::Span<const uint8_t> unpackedValues);

template <>
std::vector<uint8_t> PackUInt8Span<UInt4>(Util::Span<const uint8_t> unpackedValues);

template <>
std::vector<uint8_t> PackUInt8Span<UInt3>(Util::Span<const uint8_t> unpackedValues);

template <>
std::vector<uint8_t> PackUInt8Span<UInt2>(Util::Span<const uint8_t> unpackedValues);

template <>
std::vector<uint8_t> PackUInt8Span<UInt1>(Util::Span<const uint8_t> unpackedValues);

}  // namespace MILBlob
