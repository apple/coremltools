// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <cmath>
#include <stdint.h>
#include <type_traits>
#include <vector>

// A sub-byte type of is represented in MIL by a byte-sized struct which wraps
// an value of type IMPL_TYPE
#define DEFINE_SUB_BYTE_TYPE(NAME, IMPL_TYPE, BIT_SIZE, MASK, MAX_VAL, MIN_VAL) \
    struct NAME {                                                    \
        explicit NAME(IMPL_TYPE d);                                             \
        NAME() : data(0) {}                                                     \
        static NAME FromInt(int i);                                             \
        int GetInt() const;                                                     \
        void SetInt(int i);                                                     \
        IMPL_TYPE data;                                                         \
        static constexpr uint8_t SizeInBits = BIT_SIZE;                         \
        static constexpr uint8_t BitMask = MASK;                                \
        static constexpr IMPL_TYPE MAX = MAX_VAL;                               \
        static constexpr IMPL_TYPE MIN = MIN_VAL;                               \
        static_assert(MAX >= MIN, "Incompatible values for MIN and MAX");       \
    };

// Declares the following exports for sub-byte-type NAME
// operator ==
// operator !=
//
// Packs a sub byte vector into uint8_t representation since a vector of sub byte type
// cannot be packed.
// std::vector<uint8_t> PackSubByteVec(const std::vector<MILBlob::NAME>& vec);
//
// Unpacks a sub byte vector in uint8_t representation to a vector of the sub byte type.
// template <>
// std::vector<NAME> UnPackSubByteVec<NAME>(const std::vector<uint8_t>& vec, size_t numElements);
#define DECLARE_SUB_BYTE_TYPE_METHODS(NAME)                                            \
    bool operator==(const NAME& first, const NAME& second) noexcept;        \
    bool operator!=(const NAME& first, const NAME& second) noexcept;        \
    std::vector<uint8_t> PackSubByteVec(const std::vector<MILBlob::NAME>& vec); \
    template <>                                                                        \
    std::vector<NAME> UnPackSubByteVec<NAME>(const std::vector<uint8_t>& vec, size_t numElements);

namespace MILBlob {

template <typename T>
class IsSubByteSized {
    struct S {
        char a;
        char b;
    };
    template <typename U>
    static char Tester(decltype(&U::SizeInBits));
    template <typename U>
    static S Tester(...);

public:
    enum {
        value = sizeof(Tester<T>(0)) == sizeof(char)
    };
};

template <typename T>
constexpr bool SubByteIsByteAligned()
{
    return (8 / T::SizeInBits) * T::SizeInBits == 8;
}

template <typename T>
constexpr std::size_t SizeInBytes(std::size_t numElements)
{
    return (std::size_t)std::ceil((numElements * T::SizeInBits) / 8.0);
}

template <typename T>
std::vector<T> UnPackSubByteVec(const std::vector<uint8_t>& vec, std::size_t numElements);

DEFINE_SUB_BYTE_TYPE(Int4, int8_t, 4, 0xF, 7, -8)
DECLARE_SUB_BYTE_TYPE_METHODS(Int4)

DEFINE_SUB_BYTE_TYPE(UInt6, uint8_t, 6, 0b111111, 63, 0)
DECLARE_SUB_BYTE_TYPE_METHODS(UInt6)

DEFINE_SUB_BYTE_TYPE(UInt4, uint8_t, 4, 0xF, 15, 0)
DECLARE_SUB_BYTE_TYPE_METHODS(UInt4)

DEFINE_SUB_BYTE_TYPE(UInt3, uint8_t, 3, 0b111, 7, 0)
DECLARE_SUB_BYTE_TYPE_METHODS(UInt3)

DEFINE_SUB_BYTE_TYPE(UInt2, uint8_t, 2, 0b11, 3, 0)
DECLARE_SUB_BYTE_TYPE_METHODS(UInt2)

DEFINE_SUB_BYTE_TYPE(UInt1, uint8_t, 1, 0b1, 1, 0)
DECLARE_SUB_BYTE_TYPE_METHODS(UInt1)

}  // namespace MILBlob

namespace std {

template <>
struct hash<MILBlob::Int4> {
    size_t operator()(const MILBlob::Int4& i) const;
};

template <>
struct hash<MILBlob::UInt6> {
    size_t operator()(const MILBlob::UInt6& i) const;
};

template <>
struct hash<MILBlob::UInt4> {
    size_t operator()(const MILBlob::UInt4& i) const;
};

template <>
struct hash<MILBlob::UInt3> {
    size_t operator()(const MILBlob::UInt3& i) const;
};

template <>
struct hash<MILBlob::UInt2> {
    size_t operator()(const MILBlob::UInt2& i) const;
};

template <>
struct hash<MILBlob::UInt1> {
    size_t operator()(const MILBlob::UInt1& i) const;
};

}  // namespace std
