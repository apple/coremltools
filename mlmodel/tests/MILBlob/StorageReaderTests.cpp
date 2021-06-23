// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/StorageReader.hpp"
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

    return 0;
}

int testStorageReaderTestsDataOffset()
{
    auto tempfile = TestUtil::MakeStorageTempFileWith3Records();

    StorageReader reader(tempfile.GetFilename());

    {  // read data offset for uint8_t weights from metadata 1
        ML_ASSERT_EQ(uint64_t(128), reader.GetDataOffset(64));
    }

    {  // read data offset for Fp16 weights from metadata 2
        ML_ASSERT_EQ(uint64_t(256), reader.GetDataOffset(192));
    }

    {  // read data offset for float weights from metadata 3
        ML_ASSERT_EQ(uint64_t(384), reader.GetDataOffset(320));
    }

    return 0;
}

