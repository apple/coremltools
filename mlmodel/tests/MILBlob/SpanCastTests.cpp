// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/SubByteTypes.hpp"
#include "MILBlob/Util/SpanCast.hpp"
#include "framework/TestUtils.hpp"
#include "MLModelTests.hpp"

using namespace MILBlob::Util;


int testSpanCastTestsBasics()
{
    const std::vector<int32_t> values = {1, 2, 3, 4};
    Span<const int32_t> span = MakeSpan(values);

    {  // up cast
        Span<const int64_t> casted = SpanCast<const int64_t>(span);
        ML_ASSERT_EQ(casted.Size(), size_t(2));

        Span<const int32_t> original = SpanCast<const int32_t>(casted);
        ML_ASSERT_EQ(span.Data(), original.Data());
        ML_ASSERT_EQ(span.Size(), original.Size());
    }

    {  // down cast
        Span<const int8_t> casted = SpanCast<const int8_t>(span);
        ML_ASSERT_EQ(casted.Size(), size_t(16));

        Span<const int32_t> original = SpanCast<const int32_t>(casted);
        ML_ASSERT_EQ(span.Data(), original.Data());
        ML_ASSERT_EQ(span.Size(), original.Size());
    }

    return 0;
}

int testSpanCastTestsToInt4()
{
    std::vector<uint8_t> v;
    v.emplace_back(0x20);
    v.emplace_back(0x64);
    Span<uint8_t> span = MakeSpan(v);

    // Valid casts.
    CastToBitSpan<MILBlob::Int4>(span, 3);
    CastToBitSpan<MILBlob::Int4>(span, 4);

    // Invalid due to size being too short or too long.
    ML_ASSERT_THROWS(CastToBitSpan<MILBlob::Int4>(span, 2), std::invalid_argument);
    ML_ASSERT_THROWS(CastToBitSpan<MILBlob::Int4>(span, 5), std::invalid_argument);

    return 0;
}

int testSpanCastTestsFromInt4()
{
    std::vector<uint8_t> v;
    v.emplace_back(0x20);
    v.emplace_back(0x64);
    Span<uint8_t> span = MakeSpan(v);
    {
        Span<MILBlob::Int4> int4Span = CastToBitSpan<MILBlob::Int4>(span, 3);
        ML_ASSERT_EQ(int4Span.Size(), 3);
        Span<const uint8_t> uint8Span = CastFromBitSpan(int4Span);
        ML_ASSERT_EQ(uint8Span.Size(), 2);
    }

    return 0;
}
