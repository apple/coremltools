// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/SubByteTypes.hpp"
#include "MILBlob/Util/Span.hpp"
#include <type_traits>

namespace MILBlob {
namespace Util {

/**
    reinterpret_casts the underlying pointer in Span<SourceT> to Span<TargetT>. Callers are responsible for ensuring
    that SourceT can be interpreted as TargetT in a meaningful way as there are neither compile- nor run-time safety
    guards in place.
*/

template <typename TargetT, typename SourceT>
Span<TargetT> SpanCast(Span<SourceT> span)
{
    static_assert(!MILBlob::IsSubByteSized<TargetT>::value && !MILBlob::IsSubByteSized<SourceT>::value,
                  "SpanCast for sub-byte sized types is not supported");
    auto ptr = reinterpret_cast<TargetT*>(span.Data());
    auto size = (span.Size() * sizeof(SourceT)) / sizeof(TargetT);
    return Span<TargetT>(ptr, size);
}

/**
    Reinterpret casts the underlying Span<uint8_t> to a sub-byte type span. numElements indicates the number of
    sub-byte elements in the case where the last byte contains some padding due to round to nearest byte.
*/

template <typename TargetT, typename UINT8_T, std::enable_if_t<MILBlob::IsSubByteSized<TargetT>::value, bool> = true>
Span<TargetT> CastToBitSpan(Span<UINT8_T> span, size_t numElements)
{
    static_assert(std::is_same<UINT8_T, const uint8_t>::value || std::is_same<UINT8_T, uint8_t>::value,
                  "CastToBitSpan is only possible when casting from a uint8_t span");
    if (span.Size() != MILBlob::SizeInBytes<TargetT>(numElements)) {
        throw std::invalid_argument(
            "BitSpanCast to sub-byte type span has invalid number of elements. Sub-byte span with NumElements "
            "requires exactly Span<uint8_t>.Size() bytes.");
    }
    return Span<TargetT>((typename MILBlob::Util::voidType<UINT8_T>::type)(span.Data()), numElements);
}

/**
    Reinterpret casts the underlying sub-byte-sized Span<T> to a Span<uint8_t>
*/
template <typename SourceT, std::enable_if_t<MILBlob::IsSubByteSized<SourceT>::value, bool> = true>
Span<const uint8_t> CastFromBitSpan(Span<SourceT> span)
{
    size_t numBits = span.Size() * SourceT::SizeInBits;
    size_t numElements = numBits / 8;
    // need 1 more byte-sized element to hold remainder, if it exists
    if (numBits % 8 != 0) {
        numElements++;
    }
    return Span<const uint8_t>((const uint8_t*)span.Data(), numElements);
}

}  // namespace Util
}  // namespace MILBlob
