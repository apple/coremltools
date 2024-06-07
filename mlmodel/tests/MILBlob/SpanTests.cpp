// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/SubByteTypes.hpp"
#include "MILBlob/Util/Span.hpp"
#include "MILBlob/Util/SpanCast.hpp"
#include "MILBlob/Util/SubByteConversionUtils.hpp"
#include "framework/TestUtils.hpp"
#include "MLModelTests.hpp"

#include <experimental/type_traits>

using namespace MILBlob;
using namespace MILBlob::Util;

//----------------------------------------------------------------------
// Static-sizing interface tests
// (these will fail at compile time)
//----------------------------------------------------------------------

static_assert(!std::is_default_constructible<Span<int, 4>>::value, "Span<T, 4>() default constructor is illegal");

static_assert(!std::is_constructible<Span<int, 4>, int*, size_t>::value,
              "Span<T, 4>(ptr, size_t) constructor is illegal");

static_assert(!std::is_constructible<Span<int>, int*>::value, "Span<T, DynamicExtent>(ptr) constructor is illegal");

template <typename T>
using Has_SpanGet1_t = decltype(std::declval<T>().template Get<1>());

template <typename T>
using Has_SpanGet2_t = decltype(std::declval<T>().template Get<2>());

static_assert(!std::experimental::is_detected<Has_SpanGet1_t, Span<int>>::value,
              "Span<T, DynamicExtent>::Get<> is illegal");

static_assert(std::experimental::is_detected<Has_SpanGet1_t, Span<int, 2>>::value, "Span<T, 2>::Get<1> is legal");

static_assert(!std::experimental::is_detected<Has_SpanGet2_t, Span<int, 2>>::value, "Span<T, 2>::Get<2> is illegal");

//----------------------------------------------------------------------
// MakeSpan factory methods for std::vector
//----------------------------------------------------------------------


int testSpanTestsMakeSpanVectorMutable()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);
    static_assert(std::is_same<decltype(span)::value_type, int>::value, "MakeSpan should make non-const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());

    return 0;
}

int testSpanTestsMakeSpanVectorForcedImmutable()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan<const int>(values);
    static_assert(std::is_same<decltype(span)::value_type, decltype(span)::const_value_type>::value,
                  "MakeSpan should make const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());

    return 0;
}

int testSpanTestsMakeSpanVectorForcedImmutableFromConst()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan<const int>(values);
    static_assert(std::is_same<decltype(span)::value_type, decltype(span)::const_value_type>::value,
                  "MakeSpan should make const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());

    return 0;
}

int testSpanTestsMakeSpanVectorImmutable()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);
    static_assert(std::is_same<decltype(span)::value_type, decltype(span)::const_value_type>::value,
                  "MakeSpan should make const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());

    return 0;
}

//----------------------------------------------------------------------
// MakeSpan factory methods for std::array
//----------------------------------------------------------------------

int testSpanTestsMakeSpanArrayMutable()
{
    std::array<int, 4> values = {{1, 2, 3, 4}};
    auto span = MakeSpan(values);
    static_assert(std::is_same<decltype(span)::value_type, int>::value, "MakeSpan should make non-const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());
    ML_ASSERT_EQ(span.Get<0>(), 1);

    return 0;
}

int testSpanTestsMakeSpanArrayForcedImmutable()
{
    std::array<int, 4> values = {{1, 2, 3, 4}};
    auto span = MakeSpan<const int>(values);
    static_assert(std::is_same<decltype(span)::value_type, decltype(span)::const_value_type>::value,
                  "MakeSpan should make const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());
    ML_ASSERT_EQ(span.Get<0>(), 1);

    return 0;
}

int testSpanTestsMakeSpanArrayImmutable()
{
    const std::array<int, 4> values = {{1, 2, 3, 4}};
    auto span = MakeSpan(values);
    static_assert(std::is_same<decltype(span)::value_type, decltype(span)::const_value_type>::value,
                  "MakeSpan should make const T");

    ML_ASSERT(!span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(4));
    ML_ASSERT_EQ(span.Data(), values.data());
    ML_ASSERT_EQ(span.Get<0>(), 1);

    return 0;
}

//----------------------------------------------------------------------
// Constructors and operators
//----------------------------------------------------------------------

int testSpanTestsDefaultConstructor()
{
    Span<int> span;
    ML_ASSERT(span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(0));
    ML_ASSERT_EQ(span.Data(), nullptr);

    return 0;
}

int testSpanTestsEmpty()
{
    // test 0 length but valid ptr
    std::vector<int> v = {1, 2, 3, 4};
    Span<int> span(v.data(), 0);
    ML_ASSERT(span.IsEmpty());
    ML_ASSERT_EQ(span.Size(), size_t(0));
    ML_ASSERT_EQ(span.Data(), nullptr);

    return 0;
}

int testSpanTestsCopyAndAssignment()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    auto copied(span);
    ML_ASSERT_EQ(copied.Size(), size_t(4));

    Span<int> other;
    other = copied;
    ML_ASSERT_EQ(other.Size(), size_t(4));

    return 0;
}

int testSpanTestsImplicitConstCopyCtor()
{
    // test that a non-const span is convertible to a const span
    std::vector<int> v = {1, 2, 3, 4};
    Span<int> span(v.data(), v.size());

    {
        Span<const int> constSpan(span);
        ML_ASSERT_EQ(span.Data(), v.data());
        ML_ASSERT_EQ(span.Size(), v.size());
    }

    {
        Span<const int> constSpan(std::move(span));
        ML_ASSERT_EQ(span.Data(), v.data());
        ML_ASSERT_EQ(span.Size(), v.size());
    }

    return 0;
}

//----------------------------------------------------------------------
// Random Access
//----------------------------------------------------------------------

int testSpanTestsAccessMutable()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    ML_ASSERT_EQ(span[0], 1);
    span[0] = 10;
    ML_ASSERT_EQ(span[0], 10);

#if !defined(NDEBUG)
    ML_ASSERT_THROWS(span[5], std::range_error);
#endif

    ML_ASSERT_EQ(span.At(1), 2);
    span.At(1) = 20;
    ML_ASSERT_EQ(span.At(1), 20);

    ML_ASSERT_THROWS(span.At(5), std::range_error);

    return 0;
}

int testSpanTestsAccessImmutable()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    ML_ASSERT_EQ(span[0], 1);
    ML_ASSERT_EQ(span.At(1), 2);
    ML_ASSERT_THROWS(span.At(5), std::range_error);

    static_assert(std::is_same<decltype(span[0]), const int&>::value, "span[0] should not be mutable");
    static_assert(std::is_same<decltype(span.At(0)), const int&>::value, "span.At(0) should not be mutable");

    return 0;
}

//----------------------------------------------------------------------
// Static-sized Random Access
//----------------------------------------------------------------------

int testSpanTestsStaticSizedAccessMutable()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values).StaticResize<4>();

    ML_ASSERT_EQ(span.Get<0>(), 1);

    span.Get<0>() = 10;
    ML_ASSERT_EQ(span.Get<0>(), 10);

    return 0;
}

int testSpanTestsStaticSizedAccessImmutable()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values).StaticResize<4>();

    ML_ASSERT_EQ(span.Get<0>(), 1);

    static_assert(std::is_same<decltype(span.Get<0>()), const int&>::value, "Get<0>() should be immutable");

    return 0;
}

//----------------------------------------------------------------------
// Immutable Iteration
//----------------------------------------------------------------------

int testSpanTestsIteratorImmutable()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 0;
    for (auto& i : span) {
        static_assert(std::is_same<decltype(i), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(i, ++counter);
    }
    ML_ASSERT_EQ(counter, 4);

    return 0;
}

int testSpanTestsIteratorImmutableExplicitBeginEnd()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 0;
    for (auto itr = span.begin(); itr != span.end(); ++itr) {
        static_assert(std::is_same<decltype(*itr), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(*itr, ++counter);
    }
    ML_ASSERT_EQ(counter, 4);

    return 0;
}

int testSpanTestsIteratorImmutableExplicitCbeginCend()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 0;
    for (auto itr = span.cbegin(); itr != span.cend(); ++itr) {
        static_assert(std::is_same<decltype(*itr), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(*itr, ++counter);
    }
    ML_ASSERT_EQ(counter, 4);

    return 0;
}

int testSpanTestsIteratorImmutableExplicitRBeginREnd()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 4;
    for (auto itr = span.rbegin(); itr != span.rend(); ++itr) {
        static_assert(std::is_same<decltype(*itr), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(*itr, counter--);
    }
    ML_ASSERT_EQ(counter, 0);

    return 0;
}

int testSpanTestsIteratorImmutableExplicitCRBeginCREnd()
{
    const std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 4;
    for (auto itr = span.crbegin(); itr != span.crend(); ++itr) {
        static_assert(std::is_same<decltype(*itr), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(*itr, counter--);
    }
    ML_ASSERT_EQ(counter, 0);

    return 0;
}

//----------------------------------------------------------------------
// Mutable Iteration
//----------------------------------------------------------------------

int testSpanTestsIteratorMutable()
{
    std::vector<uint32_t> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    size_t counter = 0;
    for (auto& i : span) {
        ML_ASSERT_EQ(i, ++counter);
        i++;
        ML_ASSERT_EQ(i, counter + 1);
        ML_ASSERT_EQ(span[counter - 1], i);
    }
    ML_ASSERT_EQ(counter, size_t(4));

    return 0;
}

int testSpanTestsIteratorMutableExplicitBeginEnd()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 0;
    for (auto itr = span.begin(); itr != span.end(); ++itr) {
        ML_ASSERT_EQ(*itr, ++counter);
        (*itr)++;
        ML_ASSERT_EQ(*itr, counter + 1);
    }
    ML_ASSERT_EQ(counter, 4);

    return 0;
}

int testSpanTestsIteratorMutableExplicitCbeginCend()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 1;
    for (auto itr = span.cbegin(); itr != span.cend(); ++itr) {
        static_assert(std::is_same<decltype(*itr), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(*itr, counter++);
    }
    ML_ASSERT_EQ(counter, 5);

    return 0;
}

int testSpanTestsIteratorMutableExplicitRbeginRend()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 4;
    for (auto itr = span.rbegin(); itr != span.rend(); ++itr) {
        ML_ASSERT_EQ(*itr, counter--);
        (*itr)--;
        ML_ASSERT_EQ(*itr, counter);
    }
    ML_ASSERT_EQ(counter, 0);

    return 0;
}

int testSpanTestsIteratorMutableExplicitCRbeginCRend()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    int counter = 4;
    for (auto itr = span.crbegin(); itr != span.crend(); ++itr) {
        static_assert(std::is_same<decltype(*itr), const int&>::value, "iterator must be const");
        ML_ASSERT_EQ(*itr, counter--);
    }
    ML_ASSERT_EQ(counter, 0);

    return 0;
}

//----------------------------------------------------------------------
// Slicing
//----------------------------------------------------------------------

int testSpanTestsSlicingZeroLength()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);
    ML_ASSERT_THROWS(span.Slice(1, 0), std::range_error);

    return 0;
}

int testSpanTestsSlicingUnbounded()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    {
        auto sub = span.Slice(2);
        ML_ASSERT_EQ(sub.Size(), size_t(2));
        ML_ASSERT_EQ(sub[0], 3);
        ML_ASSERT_EQ(sub[1], 4);
    }

    {
        auto sub = span.Slice(0);
        ML_ASSERT_EQ(sub.Size(), span.Size());
        ML_ASSERT_EQ(sub.Data(), span.Data());
    }

    return 0;
}

int testSpanTestsSlicingUnboundedEdge()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    auto sub = span.Slice(3);
    ML_ASSERT_EQ(sub.Size(), size_t(1));
    ML_ASSERT_EQ(sub[0], 4);

    return 0;
}

int testSpanTestsSlicingIllegalBounds()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);
    ML_ASSERT_THROWS(span.Slice(4), std::range_error);
    ML_ASSERT_THROWS(span.Slice(0, 6), std::range_error);

    return 0;
}

int testSpanTestsSlicingBounded()
{
    std::vector<int> values = {1, 2, 3, 4};
    auto span = MakeSpan(values);

    {
        auto sub = span.Slice(1, 2);
        ML_ASSERT_EQ(sub.Size(), size_t(2));
        ML_ASSERT_EQ(sub[0], 2);
        ML_ASSERT_EQ(sub[1], 3);

        auto subsub = sub.Slice(0, 1);
        ML_ASSERT_EQ(subsub.Size(), size_t(1));
        ML_ASSERT_EQ(subsub[0], 2);
    }

    {
        auto sub = span.Slice(3, 1);
        ML_ASSERT_EQ(sub.Size(), size_t(1));
        ML_ASSERT_EQ(sub[0], 4);
    }

    {
        auto sub = span.Slice(0, span.Size());
        ML_ASSERT_EQ(sub.Size(), span.Size());
        ML_ASSERT_EQ(sub.Data(), span.Data());
    }

    return 0;
}

int testSpanTestsSlicingByDimension()
{
    std::vector<int> values = {1, 2, 3, 4, 5, 6};
    auto span = MakeSpan(values);

    {
        auto sub = span.SliceByDimension(3, 0);
        ML_ASSERT_EQ(sub.Size(), size_t(2));
        ML_ASSERT_EQ(sub[0], 1);
        ML_ASSERT_EQ(sub[1], 2);
    }

    {
        auto sub = span.SliceByDimension(3, 2);
        ML_ASSERT_EQ(sub.Size(), size_t(2));
        ML_ASSERT_EQ(sub[0], 5);
        ML_ASSERT_EQ(sub[1], 6);
    }

    return 0;
}

int testSpanTestsSlicingByInvalidDimension()
{
    std::vector<int> values = {1, 2, 3, 4, 5, 6};
    auto span = MakeSpan(values);
    ML_ASSERT_THROWS(span.SliceByDimension(4, 0), std::range_error);

    return 0;
}

int testSpanTestsSlicingByDimensionWithInvalidIndex()
{
    std::vector<int> values = {1, 2, 3, 4, 5, 6};
    auto span = MakeSpan(values);
    ML_ASSERT_THROWS(span.SliceByDimension(3, 3), std::range_error);

    return 0;
}

//----------------------------------------------------------------------
// Complex Iteration with Slicing
//----------------------------------------------------------------------

int testSpanTestsIterationIllegal()
{
    const std::vector<int> values = {1, 2, 3, 4, 5, 6};
    auto span = MakeSpan(values);
    ML_ASSERT_THROWS(span.IterateSlices(5), std::range_error);

    return 0;
}

int testSpanTestsIterationStaticSlices()
{
    const std::vector<int> values = {1, 2, 3, 4, 5, 6};
    auto span = MakeSpan(values);

    // iterate 3 values at a time
    int counter = 0;
    for (auto row : span.IterateSlices<2>()) {
        ML_ASSERT_EQ(row.Size(), size_t(2));
        ML_ASSERT_EQ(row.Get<0>(), counter + 1);
        ML_ASSERT_EQ(row.Get<1>(), counter + 2);

        counter += 2;
    }
    ML_ASSERT_EQ(counter, 6);

    return 0;
}

int testSpanTestsIterationDynamicSlices()
{
    const std::vector<int> values = {1, 2, 3, 4, 5, 6};
    auto span = MakeSpan(values);

    // iterate 3 values at a time
    int counter = 0;
    for (auto row : span.IterateSlices(3)) {
        ML_ASSERT_EQ(row.Size(), size_t(3));

        for (auto i : row) {
            ML_ASSERT_EQ(i, ++counter);
        }
    }
    ML_ASSERT_EQ(counter, 6);

    return 0;
}

int testSpanTestsIterationMultipleDims()
{
    // clang-format off
    const std::vector<int> values = {
        // shape: [2, 3, 4]
        /*0*/
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        /*1*/
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    // clang-format on

    const std::vector<int> expectedRowSums = {10, 26, 42};

    auto span = MakeSpan(values);

    for (auto span2d : span.IterateByDimension(2)) {
        ML_ASSERT_EQ(span2d.Size(), size_t(12));

        size_t rowIndex = 0;
        for (auto row : span2d.IterateByDimension(3)) {
            ML_ASSERT_EQ(row.Size(), size_t(4));

            int sum = 0;
            for (auto i : row) {
                sum += i;
            }

            ML_ASSERT_EQ(sum, expectedRowSums[rowIndex++]);
        }
    }

    return 0;
}

int testSpanTestsInt4()
{
    std::vector<uint8_t> v;
    v.emplace_back(0x20);
    v.emplace_back(0x64);
    Span<uint8_t> span = MakeSpan(v);

    Span<Int4> int4Span = CastToBitSpan<Int4>(span, 4);

    for (size_t i = 0; i < int4Span.Size(); ++i) {
        uint8_t byteBlock = ((uint8_t*)int4Span.Data())[i / 2];
        Int4 value;
        if (i % 2) {
            value.SetInt((byteBlock & 0xf0) >> 4);
        } else {
            value.SetInt(byteBlock & 0x0f);
        }
        ML_ASSERT_EQ(value, Int4(static_cast<int8_t>(i * 2)));
    }

    return 0;
}

int testSpanTestsSubbyteIntValueAt()
{
    std::vector<uint8_t> v;
    v.emplace_back(0x20);
    v.emplace_back(0x9B);
    v.emplace_back(0x08);
    Span<uint8_t> span = MakeSpan(v);

    Span<Int4> int4Span = CastToBitSpan<Int4>(span, 5);

    ML_ASSERT_EQ(int4Span.ValueAt(0), Int4(0));
    ML_ASSERT_EQ(int4Span.ValueAt(1), Int4(2));
    ML_ASSERT_EQ(int4Span.ValueAt(2), Int4(-5));
    ML_ASSERT_EQ(int4Span.ValueAt(3), Int4(-7));
    ML_ASSERT_EQ(int4Span.ValueAt(4), Int4(-8));
    ML_ASSERT_THROWS(int4Span.ValueAt(7), std::out_of_range);

    return 0;
}

int testSpanTestsSubByteUIntValueAt()
{
    {
        const std::vector<uint8_t> v = {0x20, 0x9B, 0x0F};
        Span<const uint8_t> span = MakeSpan(v);

        Span<const UInt4> uint4Span = CastToBitSpan<const UInt4>(span, 5);

        ML_ASSERT_EQ(uint4Span.ValueAt(0), UInt4(0));
        ML_ASSERT_EQ(uint4Span.ValueAt(1), UInt4(2));
        ML_ASSERT_EQ(uint4Span.ValueAt(2), UInt4(11));
        ML_ASSERT_EQ(uint4Span.ValueAt(3), UInt4(9));
        ML_ASSERT_EQ(uint4Span.ValueAt(4), UInt4(15));
        ML_ASSERT_THROWS(uint4Span.ValueAt(5), std::out_of_range);
    }

    {
        std::vector<uint8_t> v;
        v.emplace_back(0x20);
        v.emplace_back(0x9B);
        v.emplace_back(0x0F);
        Span<uint8_t> span = MakeSpan(v);

        Span<UInt2> uint2Span = CastToBitSpan<UInt2>(span, 9);

        ML_ASSERT_EQ(uint2Span.ValueAt(0), UInt2(0));
        ML_ASSERT_EQ(uint2Span.ValueAt(2), UInt2(2));
        ML_ASSERT_EQ(uint2Span.ValueAt(4), UInt2(3));
        ML_ASSERT_EQ(uint2Span.ValueAt(6), UInt2(1));
        ML_ASSERT_EQ(uint2Span.ValueAt(8), UInt2(3));
        ML_ASSERT_THROWS(uint2Span.ValueAt(9), std::out_of_range);
    }

    {
        std::vector<uint8_t> v = {0x9B};
        Span<const uint8_t> span = MakeSpan(v);

        Span<const UInt1> uint1Span = CastToBitSpan<const UInt1>(span, 8);

        ML_ASSERT_EQ(uint1Span.ValueAt(0), UInt1(1));
        ML_ASSERT_EQ(uint1Span.ValueAt(2), UInt1(0));
        ML_ASSERT_EQ(uint1Span.ValueAt(4), UInt1(1));
        ML_ASSERT_EQ(uint1Span.ValueAt(6), UInt1(0));
        ML_ASSERT_THROWS(uint1Span.ValueAt(9), std::out_of_range);
    }

    {
        // Bytes to Uint3 decimals decomposed from LSB to MSB
        // 0xFA - 1111 1010 => 11,111,010  => 7,2
        // 0x5B - 0101 1011 => 0,101,101,1 => 5,5,7
        // 0x0E - 0000 1110 => 000,011,10  => 0,3,4
        std::vector<uint8_t> v = {0xFA, 0x5B, 0x0E};
        Span<const uint8_t> span = MakeSpan(v);

        Span<const UInt3> uint3Span = CastToBitSpan<const UInt3>(span, 8);

        ML_ASSERT_EQ(uint3Span.ValueAt(0), UInt3(2));
        ML_ASSERT_EQ(uint3Span.ValueAt(2), UInt3(7));
        ML_ASSERT_EQ(uint3Span.ValueAt(4), UInt3(5));
        ML_ASSERT_EQ(uint3Span.ValueAt(5), UInt3(4));
        ML_ASSERT_THROWS(uint3Span.ValueAt(9), std::out_of_range);
    }

    return 0;
}

int testSpanTestsConstInt4()
{
    std::vector<uint8_t> v;
    v.emplace_back(0x20);
    v.emplace_back(0x64);
    Span<uint8_t> span = MakeSpan(v);

    Span<const Int4> int4Span = CastToBitSpan<const Int4>(span, 4);

    for (size_t i = 0; i < int4Span.Size(); ++i) {
        uint8_t byteBlock = ((uint8_t*)int4Span.Data())[i / 2];
        Int4 value;
        if (i % 2) {
            value.SetInt((byteBlock & 0xf0) >> 4);
        } else {
            value.SetInt(byteBlock & 0x0f);
        }
        ML_ASSERT_EQ(value, Int4(static_cast<int8_t>(i * 2)));
    }

    {
        // Test that supplied span is too small to hold the requested number of elements
        Span<const Int4> int4SpanTooSmall;
        ML_ASSERT_THROWS(int4SpanTooSmall = CastToBitSpan<const Int4>(span, 5), std::invalid_argument);
    }
    {
        // Test that supplied span is too large to hold the requested number of elements
        Span<const Int4> int4SpanTooSmall;
        ML_ASSERT_THROWS(int4SpanTooSmall = CastToBitSpan<const Int4>(span, 2), std::invalid_argument);
    }

    return 0;
}

int testSpanTestsConstUInt4()
{
    std::vector<uint8_t> v;
    v.emplace_back(0x20);
    v.emplace_back(0x64);
    Span<uint8_t> span = MakeSpan(v);

    Span<const UInt4> uint4Span = CastToBitSpan<const UInt4>(span, 4);

    for (size_t i = 0; i < uint4Span.Size(); ++i) {
        uint8_t byteBlock = ((uint8_t*)uint4Span.Data())[i / 2];
        UInt4 value;
        if (i % 2) {
            value.SetInt((byteBlock & 0xf0) >> 4);
        } else {
            value.SetInt(byteBlock & 0x0f);
        }
        ML_ASSERT_EQ(value, UInt4(static_cast<uint8_t>(i * 2)));
    }

    {
        // Test that supplied span is too small to hold the requested number of elements
        Span<const UInt4> uint4SpanTooSmall;
        ML_ASSERT_THROWS(uint4SpanTooSmall = CastToBitSpan<const UInt4>(span, 5), std::invalid_argument);
    }
    {
        // Test that supplied span is too large to hold the requested number of elements
        Span<const UInt4> uint4SpanTooLarge;
        ML_ASSERT_THROWS(uint4SpanTooLarge = CastToBitSpan<const UInt4>(span, 2), std::invalid_argument);
    }

    return 0;
}

template <typename T>
class SpanHasAtMethod {
    struct S {
        char a;
        char b;
    };
    template <typename U>
    static char Tester(decltype(&U::At));
    template <typename U>
    static S Tester(...);

public:
    enum {
        value = sizeof(Tester<T>(0)) == sizeof(char)
    };
};

int testSpanTestsSpanOverload()
{
    ML_ASSERT_EQ(true, SpanHasAtMethod<MILBlob::Util::Span<int8_t>>::value);
    ML_ASSERT_EQ(false, SpanHasAtMethod<MILBlob::Util::Span<Int4>>::value);

    ML_ASSERT_EQ(true, MILBlob::IsSubByteSized<MILBlob::Int4>::value);
    ML_ASSERT_EQ(true, MILBlob::IsSubByteSized<MILBlob::UInt4>::value);
    ML_ASSERT_EQ(true, MILBlob::IsSubByteSized<MILBlob::UInt2>::value);
    ML_ASSERT_EQ(true, MILBlob::IsSubByteSized<MILBlob::UInt1>::value);
    ML_ASSERT_EQ(true, MILBlob::IsSubByteSized<MILBlob::UInt6>::value);
    ML_ASSERT_EQ(true, MILBlob::IsSubByteSized<MILBlob::UInt3>::value);

    ML_ASSERT_EQ(true, MILBlob::SubByteIsByteAligned<MILBlob::Int4>());
    ML_ASSERT_EQ(true, MILBlob::SubByteIsByteAligned<MILBlob::UInt4>());
    ML_ASSERT_EQ(false, MILBlob::SubByteIsByteAligned<MILBlob::UInt3>());
    ML_ASSERT_EQ(false, MILBlob::SubByteIsByteAligned<MILBlob::UInt6>());

    return 0;
}
