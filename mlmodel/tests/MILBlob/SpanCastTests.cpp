// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

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

