// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Fp8.hpp"

#include <cmath>

using namespace MILBlob;

// Some global constants.
constexpr uint8_t fp32MantissaBits = 23;
constexpr int8_t fp32ExponentBias = 127;

// Helper function to handle Fp32 -> Fp8 exponent and mantissa.
template <typename FP8_TYPE, typename FP8_CAST>
void HandleFp32ToFp8ExponentMantissa(FP8_CAST& fp8, FloatCast& fp32)
{
    int32_t unbiasedExponent = fp32.components.exponent - fp32ExponentBias;
    if (unbiasedExponent + FP8_TYPE::fp8ExponentBias > 0) {
        // Normal.
        fp8.components.exponent = uint8_t(fp32.components.exponent - fp32ExponentBias + FP8_TYPE::fp8ExponentBias);
    } else {
        // Denormal.
        FloatCast fp32_bias;
        fp32_bias.components.sign = fp32.components.sign;
        fp32_bias.components.exponent = -1 * FP8_TYPE::fp8ExponentBias + fp32ExponentBias + 1;
        fp32_bias.components.mantissa = 0;
        fp32.f += fp32_bias.f;
        fp8.components.exponent = 0;
    }
    if ((fp32.components.mantissa & ((0x1 << (fp32MantissaBits - FP8_TYPE::fp8MantissaBits)) - 1)) != 0) {
        throw std::range_error("FP8 SetFloat requires rounding for the given value.");
    }
    fp8.components.mantissa = fp32.components.mantissa >> (fp32MantissaBits - FP8_TYPE::fp8MantissaBits);
}

// Helper function to handle normalizing the denormalized case for fp8.
// For denormalized fp8's, we need to normalize by subtracting a bias of 2^(1 - fp8ExponentBias)
template <typename FP8_CAST>
void HandleFp8ToFp32Denormalize(FP8_CAST& fp8, FloatCast& fp32)
{
    if (fp8.components.exponent == 0 && fp8.components.mantissa != 0) {
        fp32.components.exponent++;
        FloatCast fp32_bias;
        fp32_bias.components.sign = fp8.components.sign;
        fp32_bias.components.exponent = fp32.components.exponent;
        fp32_bias.components.mantissa = 0;
        fp32.f -= fp32_bias.f;
    }
}

// Helper function to handle exponent and mantissa for Fp8 -> Fp32 conversion.
template <typename FP8_TYPE, typename FP8_CAST>
void HandleFp8ToFp32ExponentMantissa(const FP8_CAST& fp8, FloatCast& fp32)
{
    if (fp8.components.exponent == 0 && fp8.components.mantissa == 0) {
        fp32.components.exponent = 0;
        fp32.components.mantissa = 0;
        return;
    }
    int32_t unbiasedExponent = fp8.components.exponent - FP8_TYPE::fp8ExponentBias;
    fp32.components.exponent = uint32_t(unbiasedExponent + fp32ExponentBias);
    fp32.components.mantissa =
        uint32_t(int32_t(fp8.components.mantissa << (fp32MantissaBits - FP8_TYPE::fp8MantissaBits)));
}

float Fp8E5M2::GetFloat() const
{
    FloatCast fp32 = {.f = 0};
    // Set the sign bit.
    fp32.components.sign = data.components.sign;

    // Standard NaN/Inf case. We just use the fp8 mantissa as there's
    // no strong requirements for mantissa in the NaN case.
    if (data.components.exponent == (0x1 << fp8ExponentBits) - 1) {
        fp32.components.exponent = 0xFF;
        fp32.components.mantissa = data.components.mantissa;
        return fp32.f;
    }
    HandleFp8ToFp32ExponentMantissa<Fp8E5M2, Fp8E5M2::Cast>(data, fp32);
    HandleFp8ToFp32Denormalize(data, fp32);
    return fp32.f;
}

float Fp8E4M3FN::GetFloat() const
{
    FloatCast fp32 = {.f = 0};
    // Set the sign bit.
    fp32.components.sign = data.components.sign;
    // NaN case, infinity is not supported. We just use the mantissa from the fp8.
    if (data.components.exponent == (0x1 << fp8ExponentBits) - 1 && data.components.mantissa == 0x7) {
        fp32.components.exponent = 0xFF;
        fp32.components.mantissa = data.components.mantissa;
        return fp32.f;
    }
    HandleFp8ToFp32ExponentMantissa<Fp8E4M3FN, Fp8E4M3FN::Cast>(data, fp32);
    HandleFp8ToFp32Denormalize(data, fp32);
    return fp32.f;
}

void Fp8E5M2::SetFloat(float f)
{
    FloatCast fp32 = {.f = f};
    data = {.byte = 0};
    // Set sign bit.
    data.components.sign = fp32.components.sign;

    // If f is nan or inf, set exponent to all 1's.
    if (std::isnan(f)) {
        data.components.exponent = (0x1 << fp8ExponentBits) - 1;
        data.components.mantissa = 1;
    } else if (std::isinf(f)) {
        data.components.exponent = (0x1 << fp8ExponentBits) - 1;
        data.components.mantissa = 0;
    } else if (f == 0) {
        data.components.exponent = 0;
        data.components.mantissa = 0;
    } else {
        int32_t unbiasedExponent = fp32.components.exponent - fp32ExponentBias;
        // Float is normal or denormal, check the exponent and set it.
        // For now, we throw on over/underflows. There are alternative ways to handle
        // this (round to zero).
        if (unbiasedExponent > fp8ExponentBias) {
            throw std::range_error("Fp8E5M2 SetFloat exponent overflow.");
        } else if (unbiasedExponent < (-1 * fp8ExponentBias - int32_t(fp8MantissaBits) + 1)) {
            throw std::range_error("Fp8E5M2 SetFloat exponent underflow.");
        }
        HandleFp32ToFp8ExponentMantissa<Fp8E5M2, Fp8E5M2::Cast>(data, fp32);
    }
}

void Fp8E4M3FN::SetFloat(float f)
{
    FloatCast fp32 = {.f = f};
    data = {.byte = 0};
    // Set sign bit.
    data.components.sign = fp32.components.sign;

    // If f is nan or inf, set exponent to all 1's.
    if (std::isnan(f)) {
        data.components.exponent = (0x1 << fp8ExponentBits) - 1;
        data.components.mantissa = 7;
    } else if (std::isinf(f)) {
        throw std::range_error("Fp8E4M3FN SetFloat infinity not supported.");
    } else if (f == 0) {
        data.components.exponent = 0;
        data.components.mantissa = 0;
    } else {
        int32_t unbiasedExponent = fp32.components.exponent - fp32ExponentBias;
        // Float is normal or denormal, check the exponent and set it.
        // For now, we throw on over/underflows. There are alternative ways to handle
        // this (round to zero).
        if (unbiasedExponent > fp8ExponentBias + 1) {
            throw std::range_error("Fp8E4M3FN SetFloat exponent overflow.");
        } else if (unbiasedExponent < (-1 * fp8ExponentBias - int32_t(fp8MantissaBits) + 1)) {
            // Underflow occurs when the exponent is below the minimum denormal value.
            // This means unbiased exponent is less than -fp8ExponentBias - fp8MantissaBits + 1
            throw std::range_error("Fp8E4M3FN SetFloat exponent underflow.");
        }
        HandleFp32ToFp8ExponentMantissa<Fp8E4M3FN, Fp8E4M3FN::Cast>(data, fp32);
    }
}

Fp8E5M2 Fp8E5M2::FromFloat(float f)
{
    Fp8E5M2 result;
    result.SetFloat(f);
    return result;
}

Fp8E4M3FN Fp8E4M3FN::FromFloat(float f)
{
    Fp8E4M3FN result;
    result.SetFloat(f);
    return result;
}

bool Fp8E5M2::IsNaN() const
{
    return (data.components.exponent == 0x1F && data.components.mantissa != 0);
}

bool Fp8E4M3FN::IsNaN() const
{
    return (data.components.exponent == 0xF && data.components.mantissa == 7);
}
