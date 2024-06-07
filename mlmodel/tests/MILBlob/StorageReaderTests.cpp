// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Bf16.hpp"
#include "MILBlob/Blob/BlobDataType.hpp"
#include "MILBlob/Blob/StorageReader.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/SpanCast.hpp"
#include "AutoDeleteTempFile.hpp"
#include "BlobUtils.hpp"
#include "framework/TestUtils.hpp"
#include "MLModelTests.hpp"

using namespace MILBlob;
using namespace MILBlob::Blob;
using MILBlob::TestUtil::AutoDeleteTempFile;


int testStorageReaderTestsBasicProperties()
{
    AutoDeleteTempFile tempfile;
    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_EQ(tempfile.GetFilename(), reader.GetFilename());

    return 0;
}

int testStorageReaderTestsZeroRecords()
{
    AutoDeleteTempFile tempfile;

    // clang-format off
    std::vector<uint16_t> bytes{
        // HEADER
        0x0001, 0x0000, 0x0002, 0x0000,  // count=1, version=2
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_2
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_5
        0x0000, 0x0000, 0x0000, 0x0000,  // reserved_6
        // METADATA (none)
    };
    // clang-format on

    TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<float>(64), std::range_error, "index out of bounds");

    return 0;
}

int testStorageReaderTestsTruncatedHeader()
{
    AutoDeleteTempFile tempfile;

    {
        // count=0, version=2
        std::vector<uint16_t> bytes{0x0000, 0x0000, 0x0002, 0x0000};
        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<float>(40), std::range_error, "index out of bounds");

    return 0;
}

int testStorageReaderTestsTruncatedMetadata()
{
    AutoDeleteTempFile tempfile;

    {
        // clang-format off
        std::vector<uint16_t> bytes {
            // HEADER
            0x0001, 0x0000, 0x0002, 0x0000,  // count=1, version=2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_5
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_6
            // METADATA (none)
        };
        // clang-format on

        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<float>(64), std::range_error, "index out of bounds");

    return 0;
}

int testStorageReaderTestsTruncatedData()
{
    AutoDeleteTempFile tempfile;

    {
        // clang-format off
        std::vector<uint16_t> bytes {
            // HEADER
            0x0001, 0x0000, 0x0002, 0x0000,  // count=1, version=2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_5
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_6
            // METADATA
            0xBEEF, 0xDEAD, 0x0002, 0x0000,  // sentinel=0xDEADBEEF, mil_dtype=float
            0x0009, 0x0000, 0x0000, 0x0000,  // sizeInBytes=9 bytes
            0x0080, 0x0000, 0x0000, 0x0000,  // offset
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0001, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            // DATA (8 bytes)
            0x0000, 0x0000, 0x0000, 0x0000
        };
        // clang-format on

        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<float>(64), std::range_error, "index out of bounds");

    return 0;
}

int testStorageReaderTestsIncorrectDType()
{
    AutoDeleteTempFile tempfile;

    {
        // clang-format off
        std::vector<uint16_t> bytes {
            // HEADER
            0x0001, 0x0000, 0x0002, 0x0000,  // count=1, version=2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_5
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_6
            // METADATA
            0xBEEF, 0xDEAD, 0xFFFF, 0x0000,  // sentinel=0xDEADBEEF, mil_dtype=invalid
            0x0008, 0x0000, 0x0000, 0x0000,  // sizeInBytes=8 bytes
            0x0080, 0x0000, 0x0000, 0x0000,  // offset
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0001, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            // DATA (8 bytes)
            0x0000, 0x0000, 0x0000, 0x0000
        };
        // clang-format on

        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<float>(64),
                                  std::runtime_error,
                                  "Metadata data type does not match requested type.");

    return 0;
}

int testStorageReaderTestsIncorrectMetadata()
{
    AutoDeleteTempFile tempfile;

    {
        // clang-format off
        std::vector<uint16_t> bytes {
            // HEADER
            0x0001, 0x0000, 0x0002, 0x0000,  // count=1, version=2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_5
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_6
            // METADATA
            0xBEEF, 0xDEAD, 0x000E, 0x0000,  // sentinel=0xDEADBEEF, mil_dtype=uint8_t
            0x0008, 0x0000, 0x0000, 0x0000,  // sizeInBytes=8 bytes
            0x0060, 0x0000, 0x0000, 0x0000,  // offset
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_0
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_1
            0x0001, 0x0000, 0x0000, 0x0000,  // reserved_2
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_3
            0x0000, 0x0000, 0x0000, 0x0000,  // reserved_4
            // DATA (8 bytes)
            0x0000, 0x0000, 0x0000, 0x0000
        };
        // clang-format on

        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<uint8_t>(0),
                                  std::runtime_error,
                                  "Invalid sentinel in blob_metadata.");

    return 0;
}

int testStorageReaderTestsThreeRecords()
{
    auto tempfile = TestUtil::MakeStorageTempFileWith3Records();

    StorageReader reader(tempfile.GetFilename());

    {  // read uint8_t weights from metadata 1
        auto data = reader.GetDataView<uint8_t>(64);
        ML_ASSERT_EQ(data.Size(), 5 / sizeof(uint8_t));

        std::vector<uint8_t> expectedValues = {0x02, 0x00, 0x40, 0x00, 0x07};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }

    {  // read Fp16 weights from metadata 2
        auto data = reader.GetDataView<Fp16>(192);
        ML_ASSERT_EQ(data.Size(), 8 / sizeof(Fp16));

        std::vector<Fp16> expectedValues = {Fp16(0x000E), Fp16(0xC0FE), Fp16(0x0810), Fp16(0x0000)};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }

    {  // read float weights from metadata 3
        auto data = reader.GetDataView<float>(320);
        ML_ASSERT_EQ(data.Size(), 16 / sizeof(float));

        std::vector<uint32_t> expectedValues = {0x700000, 0xC0FEE, 0x8FACE, 0x91FADE};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<float>(Util::MakeSpan(expectedValues)));
    }

    {  // read Bf16 weights from metadata 4
        auto data = reader.GetDataView<Bf16>(576);
        ML_ASSERT_EQ(data.Size(), 8 / sizeof(Bf16));

        std::vector<Bf16> expectedValues = {Bf16(0x000E), Bf16(0xC0FE), Bf16(0x0810), Bf16(0x0000)};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }

    {  // read int16_t weights from metadata 5
        auto data = reader.GetDataView<int16_t>(704);
        ML_ASSERT_EQ(data.Size(), 4 / sizeof(int16_t));

        std::vector<int16_t> expectedValues = {int16_t(0xe8d0), int16_t(0x007e)};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }

    {  // read uint16_t weights from metadata 6
        auto data = reader.GetDataView<uint16_t>(832);
        ML_ASSERT_EQ(data.Size(), 4 / sizeof(uint16_t));

        std::vector<uint16_t> expectedValues = {uint16_t(0xe8d0), uint16_t(0x007e)};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }

    {  // read Int4 weights from metadata t
        auto int4Data = reader.GetDataView<Int4>(960);
        ML_ASSERT_EQ(int4Data.Size(), 8);
        auto uint8Data = MILBlob::Util::CastFromBitSpan(int4Data);
        auto data = MILBlob::Util::SpanCast<const uint16_t>(uint8Data);

        std::vector<uint16_t> expectedValues = {uint16_t(0xe8d0), uint16_t(0x007e)};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }
    {
        auto uint3Data = reader.GetDataView<UInt3>(1472);
        ML_ASSERT_EQ(uint3Data.Size(), 9);
        auto uint8Data = MILBlob::Util::CastFromBitSpan(uint3Data);
        auto data = MILBlob::Util::SpanCast<const uint16_t>(uint8Data);

        std::vector<uint16_t> expectedValues = {uint16_t(0xEC24), uint16_t(0x1D45)};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }
    {
        auto int32Data = reader.GetDataView<int32_t>(1600);
        ML_ASSERT_EQ(int32Data.Size(), 1);
        std::vector<int32_t> expectedValues = {int32_t(0x0C24)};
        ML_ASSERT_SPAN_EQ(int32Data, Util::MakeSpan(expectedValues));
    }
    {
        auto uint32Data = reader.GetDataView<uint32_t>(1728);
        ML_ASSERT_EQ(uint32Data.Size(), 2);
        std::vector<uint32_t> expectedValues = {uint32_t(0x0C24), uint32_t(0xBEEF)};
        ML_ASSERT_SPAN_EQ(uint32Data, Util::MakeSpan(expectedValues));
    }
    {
        auto fp8E5M2Data = reader.GetDataView<Fp8E5M2>(1856);
        ML_ASSERT_EQ(fp8E5M2Data.Size(), 4);
        std::vector<Fp8E5M2> expectedValues = {Fp8E5M2(0xBE), Fp8E5M2(0xEF), Fp8E5M2(0x00), Fp8E5M2(0xCA)};
        ML_ASSERT_SPAN_EQ(fp8E5M2Data, Util::MakeSpan(expectedValues));
    }
    {
        auto fp8E4M3FNData = reader.GetDataView<Fp8E4M3FN>(1984);
        ML_ASSERT_EQ(fp8E4M3FNData.Size(), 4);
        std::vector<Fp8E4M3FN> expectedValues = {Fp8E4M3FN(0xBE), Fp8E4M3FN(0xEF), Fp8E4M3FN(0x00), Fp8E4M3FN(0xCA)};
        ML_ASSERT_SPAN_EQ(fp8E4M3FNData, Util::MakeSpan(expectedValues));
    }

    return 0;
}

int testStorageReaderTestsRawData()
{
    auto tempfile = TestUtil::MakeStorageTempFileWith3Records();

    StorageReader reader(tempfile.GetFilename());

    {  // read uint8_t weights from metadata 1
        const auto data = reader.GetRawDataView(64);
        ML_ASSERT_EQ(data.Size(), size_t(5));

        std::vector<uint8_t> expectedValues = {0x02, 0x00, 0x40, 0x00, 0x07};
        ML_ASSERT_SPAN_EQ(data, Util::MakeSpan(expectedValues));
    }

    {  // read Fp16 weights from metadata 2
        auto data = reader.GetRawDataView(192);
        ML_ASSERT_EQ(data.Size(), size_t(8));

        std::vector<Fp16> expectedValues = {Fp16(0x000E), Fp16(0xC0FE), Fp16(0x0810), Fp16(0x0000)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValues)));
    }

    {  // read float weights from metadata 3
        auto data = reader.GetRawDataView(320);
        ML_ASSERT_EQ(data.Size(), size_t(16));

        std::vector<uint32_t> expectedValues = {0x700000, 0xC0FEE, 0x8FACE, 0x91FADE};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<uint8_t>(Util::MakeSpan(expectedValues)));
    }

    {  // read Bf16 weights from metadata 5
        auto data = reader.GetRawDataView(576);
        ML_ASSERT_EQ(data.Size(), size_t(8));

        std::vector<Bf16> expectedValues = {Bf16(0x000E), Bf16(0xC0FE), Bf16(0x0810), Bf16(0x0000)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValues)));
    }

    {  // read int16_t weights from metadata 6
        auto data = reader.GetRawDataView(704);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        std::vector<int16_t> expectedValue = {int16_t(0xe8d0), int16_t(0x7e)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }

    {  // read uint16_t weights from metadata 7
        auto data = reader.GetRawDataView(832);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        std::vector<uint16_t> expectedValue = {uint16_t(0xe8d0), uint16_t(0x7e)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }

    {  // read Int4 weights from metadata 8
        auto data = reader.GetRawDataView(960);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        // remember int4's are actually stored here, so this vector type is immaterial
        // (cant materialize an int4 span from an int4 vector)
        std::vector<uint16_t> expectedValue = {uint16_t(0xe8d0), uint16_t(0x7e)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }

    {  // read UInt4 weights from metadata 9
        auto data = reader.GetRawDataView(1088);
        ML_ASSERT_EQ(data.Size(), size_t(3));

        std::vector<uint8_t> expectedValue = {uint8_t(0xd1), uint8_t(0xe8), uint8_t(0x7c)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }
    {  // read UInt1 weights from metadata 10
        auto data = reader.GetRawDataView(1216);
        ML_ASSERT_EQ(data.Size(), size_t(3));

        std::vector<uint8_t> expectedValue = {uint8_t(0x24), uint8_t(0xec), uint8_t(0xf7)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }
    {  // read UInt2 weights from metadata 11
        auto data = reader.GetRawDataView(1344);
        ML_ASSERT_EQ(data.Size(), size_t(2));

        std::vector<uint8_t> expectedValue = {uint8_t(0x24), uint8_t(0xec)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }

    {  // read UInt3 weights from metadata 12
        auto data = reader.GetRawDataView(1472);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        std::vector<uint8_t> expectedValue = {uint8_t(0x24), uint8_t(0xEC), uint8_t(0x45), uint8_t(0x1D)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }
    {  // read Int32 weights from metadata 13
        auto data = reader.GetRawDataView(1600);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        std::vector<uint8_t> expectedValue = {uint8_t(0x24), uint8_t(0x0C), uint8_t(0x00), uint8_t(0x00)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }
    {  // read Uint32 weights from metadata 14
        auto data = reader.GetRawDataView(1728);
        ML_ASSERT_EQ(data.Size(), size_t(8));

        std::vector<uint8_t> expectedValue = {uint8_t(0x24),
                                              uint8_t(0x0C),
                                              uint8_t(0x00),
                                              uint8_t(0x00),
                                              uint8_t(0xEF),
                                              uint8_t(0xBE),
                                              uint8_t(0x00),
                                              uint8_t(0x00)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }
    {  // read Fp8E5M2 weights from metadata 15
        auto data = reader.GetRawDataView(1856);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        std::vector<uint8_t> expectedValue = {uint8_t(0xBE), uint8_t(0xEF), uint8_t(0x00), uint8_t(0xCA)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }
    {  // read Fp8E4M3FN weights from metadata 16
        auto data = reader.GetRawDataView(1984);
        ML_ASSERT_EQ(data.Size(), size_t(4));

        std::vector<uint8_t> expectedValue = {uint8_t(0xBE), uint8_t(0xEF), uint8_t(0x00), uint8_t(0xCA)};
        ML_ASSERT_SPAN_EQ(data, Util::SpanCast<const uint8_t>(Util::MakeSpan(expectedValue)));
    }

    return 0;
}

int testStorageReaderTestsDataOffset()
{
    auto tempfile = TestUtil::MakeStorageTempFileWith3Records();

    StorageReader reader(tempfile.GetFilename());

    {  // read data offset for uint8_t weights from metadata 1
        ML_ASSERT_EQ(uint64_t(128), reader.GetDataOffset(64));
        ML_ASSERT_EQ(uint64_t(5), reader.GetDataSize(64));
        ML_ASSERT_EQ(BlobDataType::UInt8, reader.GetDataType(64));
    }

    {  // read data offset for Fp16 weights from metadata 2
        ML_ASSERT_EQ(uint64_t(256), reader.GetDataOffset(192));
        ML_ASSERT_EQ(uint64_t(8), reader.GetDataSize(192));
        ML_ASSERT_EQ(BlobDataType::Float16, reader.GetDataType(192));
    }

    {  // read data offset for float weights from metadata 3
        ML_ASSERT_EQ(uint64_t(384), reader.GetDataOffset(320));
        ML_ASSERT_EQ(uint64_t(16), reader.GetDataSize(320));
        ML_ASSERT_EQ(BlobDataType::Float32, reader.GetDataType(320));
    }

    {  // read data offset for Bf16 weights from metadata 4
        ML_ASSERT_EQ(uint64_t(640), reader.GetDataOffset(576));
        ML_ASSERT_EQ(uint64_t(8), reader.GetDataSize(576));
        ML_ASSERT_EQ(BlobDataType::BFloat16, reader.GetDataType(576));
    }

    {  // read data offset for UInt1 weights from metadata 9
        ML_ASSERT_EQ(uint64_t(1280), reader.GetDataOffset(1216));
        ML_ASSERT_EQ(uint64_t(3), reader.GetDataSize(1216));
        ML_ASSERT_EQ(BlobDataType::UInt1, reader.GetDataType(1216));
        ML_ASSERT_EQ(7, reader.GetDataPaddingInBits(1216));
    }

    {  // read data offset for UInt3 weights from metadata 12
        ML_ASSERT_EQ(uint64_t(0x600), reader.GetDataOffset(1472));
        ML_ASSERT_EQ(uint64_t(4), reader.GetDataSize(1472));
        ML_ASSERT_EQ(BlobDataType::UInt3, reader.GetDataType(1472));
        ML_ASSERT_EQ(5, reader.GetDataPaddingInBits(1472));
    }

    {  // read data offset for Int32 weights from metadata 13
        ML_ASSERT_EQ(uint64_t(1664), reader.GetDataOffset(1600));
        ML_ASSERT_EQ(uint64_t(4), reader.GetDataSize(1600));
        ML_ASSERT_EQ(BlobDataType::Int32, reader.GetDataType(1600));
        ML_ASSERT_EQ(0, reader.GetDataPaddingInBits(1600));
    }

    {  // read data offset for UInt32 weights from metadata 14
        ML_ASSERT_EQ(uint64_t(1792), reader.GetDataOffset(1728));
        ML_ASSERT_EQ(uint64_t(8), reader.GetDataSize(1728));
        ML_ASSERT_EQ(BlobDataType::UInt32, reader.GetDataType(1728));
        ML_ASSERT_EQ(0, reader.GetDataPaddingInBits(1728));
    }

    {  // read data offset for Fp8E5M2 weights from metadata 15
        ML_ASSERT_EQ(uint64_t(1920), reader.GetDataOffset(1856));
        ML_ASSERT_EQ(uint64_t(4), reader.GetDataSize(1856));
        ML_ASSERT_EQ(BlobDataType::Float8E5M2, reader.GetDataType(1856));
        ML_ASSERT_EQ(0, reader.GetDataPaddingInBits(1856));
    }

    {  // read data offset for Fp8E4M3FN weights from metadata 15
        ML_ASSERT_EQ(uint64_t(1920), reader.GetDataOffset(1984));
        ML_ASSERT_EQ(uint64_t(4), reader.GetDataSize(1984));
        ML_ASSERT_EQ(BlobDataType::Float8E4M3FN, reader.GetDataType(1984));
        ML_ASSERT_EQ(0, reader.GetDataPaddingInBits(1984));
    }

    return 0;
}

int testStorageReaderTestsInt8Data()
{
    AutoDeleteTempFile tempfile;
    const std::vector<int8_t> data{1, -1, -20, 25, 13};
    uint64_t offset = 0;
    {
        StorageWriter writer(tempfile.GetFilename());
        auto span = Util::MakeSpan(data);
        offset = writer.WriteData(span);
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_EQ(reader.GetDataType(offset), BlobDataType::Int8);
    const auto readData = reader.GetDataView<int8_t>(offset);
    ML_ASSERT_EQ(readData.Size(), data.size());

    ML_ASSERT_SPAN_EQ(readData, Util::MakeSpan(data));

    return 0;
}

int testStorageReaderTestsAllOffsets()
{
    AutoDeleteTempFile tempfile;
    const std::vector<std::vector<int8_t>> dataVectors = {{1, -1, -20, 25, 13},
                                                          {2, -2, -40, 50, 26},
                                                          {3, -3, -60, 75, 39}};
    std::vector<uint64_t> originalOffsets;
    originalOffsets.reserve(3);
    {
        StorageWriter writer(tempfile.GetFilename());
        for (size_t i = 0; i < dataVectors.size(); ++i) {
            auto span = Util::MakeSpan(dataVectors[i]);
            originalOffsets.push_back(writer.WriteData(span));
        }
    }

    StorageReader reader(tempfile.GetFilename());
    auto obtainedOffsets = reader.GetAllOffsets();
    ML_ASSERT_EQ(obtainedOffsets.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        ML_ASSERT_EQ(obtainedOffsets[i], originalOffsets[i]);
        const auto readData = reader.GetDataView<int8_t>(obtainedOffsets[i]);
        ML_ASSERT_EQ(readData.Size(), dataVectors[i].size());
        ML_ASSERT_SPAN_EQ(readData, Util::MakeSpan(dataVectors[i]));
    }

    return 0;
}

int testStorageReaderTestsAllOffsetsWithEmptyBlobFile()
{
    AutoDeleteTempFile tempfile;
    {
        StorageWriter writer(tempfile.GetFilename());
    }

    StorageReader reader(tempfile.GetFilename());
    auto obtainedOffsets = reader.GetAllOffsets();
    ML_ASSERT_EQ(obtainedOffsets.size(), 0);

    return 0;
}

int testStorageReaderTestsIsEncryptedWithEmptyBlobFile()
{
    AutoDeleteTempFile tempfile;
    {
        StorageWriter writer(tempfile.GetFilename());
    }

    StorageReader reader(tempfile.GetFilename());
    ML_ASSERT_NOT(reader.IsEncrypted());

    return 0;
}
