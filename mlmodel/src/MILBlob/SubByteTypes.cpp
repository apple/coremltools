// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Util/Verify.hpp"

#include "MILBlob/SubByteTypes.hpp"
#include "MILBlob/Util/SubByteConversionUtils.hpp"
#include <string>
#include <vector>

namespace MILBlob {

struct IndexAndOffset {
    uint64_t index;
    uint8_t offset;
};

static IndexAndOffset GetIndexAndOffsetForSubByteValue(uint64_t i, uint8_t numBits)
{
    IndexAndOffset ret;

    uint64_t startBit = numBits * i;

    ret.index = startBit / 8;
    ret.offset = startBit % 8;

    return ret;
}

template <typename T>
std::vector<uint8_t> PackSubByteVecForNonByteAligned(Util::Span<const decltype(T::data)> span)
{
    std::vector<uint8_t> ret(MILBlob::SizeInBytes<T>(span.Size()), 0);

    for (uint64_t i = 0; i < span.Size(); i++) {
        MILVerifyIsTrue(span[i] <= T::MAX && span[i] >= T::MIN,
                        std::range_error,
                        "Value " + std::to_string(span[i]) + " is outside allowed subbyte datatype range [" +
                            std::to_string(T::MIN) + ", " + std::to_string(T::MAX) + "].");

        auto indexAndOffset = GetIndexAndOffsetForSubByteValue(i, T::SizeInBits);
        auto idx = indexAndOffset.index;
        auto offset = indexAndOffset.offset;

        ret[idx] |= ((uint8_t)(span[i] << offset));
        if (offset > 8 - T::SizeInBits) {
            // part of the i'th element of span spills over to idx+1
            // uint8_t rshift = T::SizeInBits - (8 - offset);
            uint8_t rshift = 8 - offset;
            ret[idx + 1] |= ((uint8_t)span[i] >> rshift);
        }
    }

    return ret;
}

template <typename T>
std::vector<uint8_t> PackSubByteVecImpl(Util::Span<const decltype(T::data)> vec)
{
    if constexpr (!MILBlob::SubByteIsByteAligned<T>()) {
        return PackSubByteVecForNonByteAligned<T>(vec);
    }
    const auto ElementsPerByte = 8 / T::SizeInBits;
    std::vector<uint8_t> ret(MILBlob::SizeInBytes<T>(vec.Size()));
    for (size_t i = 0; i < vec.Size(); i++) {
        size_t shiftAmmount = T::SizeInBits * (i % ElementsPerByte);
        MILVerifyIsTrue(vec[i] <= T::MAX && vec[i] >= T::MIN,
                        std::range_error,
                        "Value " + std::to_string(vec[i]) + " is outside allowed subbyte datatype range [" +
                            std::to_string(T::MIN) + ", " + std::to_string(T::MAX) + "].");
        ret[i / ElementsPerByte] |= (static_cast<uint8_t>((vec[i] & T::BitMask) << shiftAmmount));
    }
    return ret;
}

#define DEFINE_PACK_SUB_BYTE_VEC(TYPE)                                                              \
    std::vector<uint8_t> PackSubByteVec(const std::vector<TYPE>& vec)                               \
    {                                                                                               \
        using impl_t = decltype(TYPE::data);                                                        \
        Util::Span<const impl_t> int8Span(reinterpret_cast<const impl_t*>(vec.data()), vec.size()); \
        return PackSubByteVecImpl<TYPE>(int8Span);                                                  \
    }

#define DECLARE_SUB_BYTE_TYPE(TYPE_NAME) DEFINE_PACK_SUB_BYTE_VEC(TYPE_NAME)
#include "MILBlob/SubByteTypeList.hpp"
#undef DECLARE_SUB_BYTE_TYPE

#define DEFINE_UNPACK_SUB_BYTE_VEC(TYPE)                                                          \
    template <>                                                                                   \
    std::vector<TYPE> UnPackSubByteVec<TYPE>(const std::vector<uint8_t>& vec, size_t numElements) \
    {                                                                                             \
        return UnPackSubByteVecImpl<TYPE>(vec, numElements);                                      \
    }

template <typename T>
std::vector<T> UnPackSubByteVecImpl(const std::vector<uint8_t>& vec, size_t numElements)
{
    std::vector<T> ret(numElements);
    MILVerifyIsTrue(
        vec.size() == MILBlob::SizeInBytes<T>(numElements),
        std::invalid_argument,
        "Unpacking to sub-byte type vector has invalid number of elements. Sub-byte vector with NumElements "
        "requires exactly vec.size() bytes.");
    Util::Span<T> subByteSpan((typename MILBlob::Util::voidType<T>::type)(vec.data()), numElements);
    for (size_t i = 0; i < numElements; i++) {
        ret[i] = subByteSpan.ValueAt(i);
    }
    return ret;
}

#define DECLARE_SUB_BYTE_TYPE(TYPE_NAME) DEFINE_UNPACK_SUB_BYTE_VEC(TYPE_NAME)
#include "MILBlob/SubByteTypeList.hpp"
#undef DECLARE_SUB_BYTE_TYPE

template <>
std::vector<uint8_t> PackInt8Span<Int4>(Util::Span<const int8_t> unpackedValues)
{
    return PackSubByteVecImpl<Int4>(unpackedValues);
}

template <>
std::vector<uint8_t> PackUInt8Span<UInt6>(Util::Span<const uint8_t> unpackedValues)
{
    return PackSubByteVecImpl<UInt6>(unpackedValues);
}

template <>
std::vector<uint8_t> PackUInt8Span<UInt4>(Util::Span<const uint8_t> unpackedValues)
{
    return PackSubByteVecImpl<UInt4>(unpackedValues);
}

template <>
std::vector<uint8_t> PackUInt8Span<UInt3>(Util::Span<const uint8_t> unpackedValues)
{
    return PackSubByteVecImpl<UInt3>(unpackedValues);
}

template <>
std::vector<uint8_t> PackUInt8Span<UInt2>(Util::Span<const uint8_t> unpackedValues)
{
    return PackSubByteVecImpl<UInt2>(unpackedValues);
}

template <>
std::vector<uint8_t> PackUInt8Span<UInt1>(Util::Span<const uint8_t> unpackedValues)
{
    return PackSubByteVecImpl<UInt1>(unpackedValues);
}

// Class methods for Int4, UInt4, etc.
#define IMPLEMENT_METHODS_FOR_SUB_BYTE_TYPE(TYPE_NAME)                        \
    TYPE_NAME::TYPE_NAME(decltype(TYPE_NAME::data) d)                         \
    {                                                                         \
        MILVerifyIsTrue(d <= TYPE_NAME::MAX && d >= TYPE_NAME::MIN,           \
                        std::range_error,                                     \
                        #TYPE_NAME " value is out of range.");                \
        data = d;                                                             \
    }                                                                         \
    /* static */ TYPE_NAME TYPE_NAME::FromInt(int i)                          \
    {                                                                         \
        TYPE_NAME result;                                                     \
        result.SetInt(i);                                                     \
        return result;                                                        \
    }                                                                         \
    int TYPE_NAME::GetInt() const                                             \
    {                                                                         \
        return static_cast<int>(data);                                        \
    }                                                                         \
    void TYPE_NAME::SetInt(int i)                                             \
    {                                                                         \
        MILVerifyIsTrue(i <= TYPE_NAME::MAX && i >= TYPE_NAME::MIN,           \
                        std::range_error,                                     \
                        #TYPE_NAME " value is out of range.");                \
        data = static_cast<decltype(TYPE_NAME::data)>(i);                     \
        return;                                                               \
    }                                                                         \
    bool operator==(const TYPE_NAME& first, const TYPE_NAME& second) noexcept \
    {                                                                         \
        return first.data == second.data;                                     \
    }                                                                         \
    bool operator!=(const TYPE_NAME& first, const TYPE_NAME& second) noexcept \
    {                                                                         \
        return first.data != second.data;                                     \
    }                                                                         \
    static_assert(sizeof(TYPE_NAME) == 1, #TYPE_NAME " struct must be of size 1 byte");

#define DECLARE_SUB_BYTE_TYPE(TYPE_NAME) IMPLEMENT_METHODS_FOR_SUB_BYTE_TYPE(TYPE_NAME)
#include "MILBlob/SubByteTypeList.hpp"
#undef DECLARE_SUB_BYTE_TYPE

};  // namespace MILBlob

namespace std {

// +128 here so that casting i.data to size_t, for T==Int4, is safe
#define DEFINE_HASH_FOR_SUB_BYTE_TYPE(TYPE)                      \
    size_t hash<MILBlob::TYPE>::operator()(const MILBlob::TYPE& i) const \
    {                                                            \
        return static_cast<size_t>(i.data + 128);                \
    }

#define DECLARE_SUB_BYTE_TYPE(TYPE_NAME) DEFINE_HASH_FOR_SUB_BYTE_TYPE(TYPE_NAME)
#include "MILBlob/SubByteTypeList.hpp"
#undef DECLARE_SUB_BYTE_TYPE

}  // namespace std
