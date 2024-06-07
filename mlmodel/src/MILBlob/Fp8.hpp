// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>

namespace MILBlob {

// General helper typedef to help process an FP32 in different forms/its
// constituent components.
typedef union {
    float f;
    uint32_t bytes;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } components;
} FloatCast;

// Macro for FP8 types.
#define DECLARE_FP8_TYPE(NAME, EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS)                                        \
    struct NAME {                                                                                       \
        typedef union {                                                                                            \
            uint8_t byte;                                                                                          \
            struct {                                                                                               \
                uint8_t mantissa : MANTISSA_BITS;                                                                  \
                uint8_t exponent : EXPONENT_BITS;                                                                  \
                uint8_t sign : 1;                                                                                  \
            } components;                                                                                          \
        } Cast;                                                                                                    \
        explicit NAME(uint8_t d)                                                                                   \
        {                                                                                                          \
            data.byte = d;                                                                                         \
        };                                                                                                         \
        NAME()                                                                                                     \
        {                                                                                                          \
            data.byte = 0;                                                                                         \
        }                                                                                                          \
        static NAME FromFloat(float f);                                                                            \
        float GetFloat() const;                                                                                    \
        void SetFloat(float f);                                                                                    \
        uint8_t GetByte() const                                                                                    \
        {                                                                                                          \
            return data.byte;                                                                                      \
        }                                                                                                          \
        void SetByte(uint8_t byte)                                                                                 \
        {                                                                                                          \
            data.byte = byte;                                                                                      \
        }                                                                                                          \
        bool IsNaN() const;                                                                                        \
        Cast data;                                                                                                 \
        static constexpr int8_t fp8ExponentBias = EXPONENT_BIAS;                                                   \
        static constexpr uint8_t fp8ExponentBits = EXPONENT_BITS;                                                  \
        static constexpr uint8_t fp8MantissaBits = MANTISSA_BITS;                                                  \
        static_assert(fp8ExponentBits + fp8MantissaBits == 7, "Number of exponent and mantissa bits should be 7"); \
    };                                                                                                             \
    inline bool operator==(const NAME& first, const NAME& second) noexcept                                         \
    {                                                                                                              \
        if ((first.data.byte & 0x7F) == 0 && (second.data.byte & 0x7F) == 0) {                                     \
            return true;                                                                                           \
        }                                                                                                          \
        if (first.IsNaN() && second.IsNaN()) {                                                                     \
            return false;                                                                                          \
        }                                                                                                          \
        return first.data.byte == second.data.byte;                                                                \
    }                                                                                                              \
    inline bool operator!=(const NAME& first, const NAME& second) noexcept                                         \
    {                                                                                                              \
        if ((first.data.byte & 0x7F) == 0 && (second.data.byte & 0x7F) == 0) {                                     \
            return false;                                                                                          \
        }                                                                                                          \
        if (first.IsNaN() && second.IsNaN()) {                                                                     \
            return true;                                                                                           \
        }                                                                                                          \
        return first.data.byte != second.data.byte;                                                                \
    }

// Define the types.
DECLARE_FP8_TYPE(Fp8E5M2, 5, 2, 15)
DECLARE_FP8_TYPE(Fp8E4M3FN, 4, 3, 7)

}  // namespace MILBlob

namespace std {

template <>
struct hash<MILBlob::Fp8E5M2> {
    size_t operator()(const MILBlob::Fp8E5M2& fp) const
    {
        return fp.data.byte;
    }
};

template <>
struct hash<MILBlob::Fp8E4M3FN> {
    size_t operator()(const MILBlob::Fp8E4M3FN& fp) const
    {
        return fp.data.byte;
    }
};

}  // namespace std
