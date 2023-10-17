// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Util/Span.hpp"
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

