// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Bf16.hpp"
#include "MILBlob/Blob/StorageFormat.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Fp16.hpp"
#include "AutoDeleteTempFile.hpp"
#include "BlobUtils.hpp"
#include "framework/TestUtils.hpp"
#include "MLModelTests.hpp"

#include <stdexcept>
#include <string_view>

using namespace MILBlob;
using namespace MILBlob::Blob;
using MILBlob::TestUtil::AutoDeleteTempFile;


namespace {

[[nodiscard]] bool IsCorrectHeader(const std::string& filePath, uint32_t count)
{
    storage_header header;
    TestUtil::ReadData(filePath, header, 0);
    return header.count == count && header.version == uint32_t(2);
}

template <typename T>
[[nodiscard]] bool IsCorrectMetadata(const std::string& filePath,
                                     uint64_t offset,
                                     uint64_t entryCount,
                                     BlobDataType dataType)
{
    blob_metadata metadata;
    TestUtil::ReadData<blob_metadata>(filePath, metadata, offset);

    return metadata.sentinel == BlobMetadataSentinel && metadata.mil_dtype == dataType &&
           metadata.sizeInBytes == entryCount * sizeof(T) && metadata.offset % DefaultStorageAlignment == 0;
}

template <typename T>
[[nodiscard]] bool IsCorrectData(const std::string& filePath, uint64_t offset, Util::Span<const T> expectedSpan)
{
    blob_metadata metadata;
    TestUtil::ReadData<blob_metadata>(filePath, metadata, offset);

    std::vector<T> output(expectedSpan.Size());
    auto outputSpan = Util::MakeSpan(output);
    TestUtil::ReadBlobFile<T>(filePath, metadata.offset, outputSpan);

    return outputSpan.Size() == expectedSpan.Size() &&
           std::equal(outputSpan.begin(), outputSpan.end(), expectedSpan.begin());
}

}  // anonymous namespace

int testStorageWriterTestsSupportedTypes()
{
    AutoDeleteTempFile tempfile;
    auto filePath = tempfile.GetFilename();
    uint32_t headerCount = 0;

    // Writing uint8_t values
    {
        const std::vector<uint8_t> val = {0x01, 0xde, 0xad, 0x10};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename());
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount /*count*/));
        ML_ASSERT(IsCorrectMetadata<uint8_t>(filePath, offset, 4, BlobDataType::UInt8));
        ML_ASSERT(IsCorrectData<uint8_t>(filePath, offset, expectedSpan));
    }

    // Writing uint16_t values
    {
        const std::vector<uint16_t> val = {0xFFC2, 0x0, 0x8000, 0x03DE};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount));
        ML_ASSERT(IsCorrectMetadata<uint16_t>(filePath, offset, 4, BlobDataType::UInt16));
        ML_ASSERT(IsCorrectData<uint16_t>(filePath, offset, expectedSpan));
    }

    // Writing int16_t values
    {
        const std::vector<int16_t> val = {int16_t(0xFFC2), 0x7FFF, int16_t(0x8000), 0x03DE};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount));
        ML_ASSERT(IsCorrectMetadata<int16_t>(filePath, offset, 4, BlobDataType::Int16));
        ML_ASSERT(IsCorrectData<int16_t>(filePath, offset, expectedSpan));
    }

    // Writing bf16 values
    {
        const std::vector<Bf16> val = {Bf16(0x12), Bf16(0x00), Bf16(0x124), Bf16(0xabcd)};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount /*count*/));
        ML_ASSERT(IsCorrectMetadata<Bf16>(filePath, offset, 4, BlobDataType::BFloat16));
        ML_ASSERT(IsCorrectData<Bf16>(filePath, offset, expectedSpan));
    }

    // Writing fp16 values
    {
        const std::vector<Fp16> val = {Fp16(0x12), Fp16(0x00), Fp16(0x124), Fp16(0xabcd)};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount /*count*/));
        ML_ASSERT(IsCorrectMetadata<Fp16>(filePath, offset, 4, BlobDataType::Float16));
        ML_ASSERT(IsCorrectData<Fp16>(filePath, offset, expectedSpan));
    }

    // Writing fp32 values
    {
        const std::vector<float> val = {3.14f, 72.1535155f, 0xabcde, 0x007};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount /*count*/));
        ML_ASSERT(IsCorrectMetadata<float>(filePath, offset, 4, BlobDataType::Float32));
        ML_ASSERT(IsCorrectData<float>(filePath, offset, expectedSpan));
    }

    // Writing int8 values
    {
        const std::vector<int8_t> val = {1, -1, 10, -25};
        auto expectedSpan = Util::MakeSpan(val);
        uint64_t offset = 0;
        {
            StorageWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            offset = writer.WriteData(expectedSpan);
        }

        ML_ASSERT_EQ(offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT(IsCorrectHeader(filePath, ++headerCount /*count*/));
        ML_ASSERT(IsCorrectMetadata<int8_t>(filePath, offset, 4, BlobDataType::Int8));
        ML_ASSERT(IsCorrectData<int8_t>(filePath, offset, expectedSpan));
    }

    return 0;
}

int testStorageWriterTestsAppendToExistingFile()
{
    // File does not exists, creates one
    {
        AutoDeleteTempFile tempfile;
        StorageWriter(tempfile.GetFilename(), /* truncateFile */ false);
        storage_header header;
        TestUtil::ReadData(tempfile.GetFilename(), header, 0);
        ML_ASSERT_EQ(header.count, uint32_t(0));
        ML_ASSERT_EQ(header.version, uint32_t(2));
    }

    // Empty file exists, adds header
    {
        AutoDeleteTempFile tempfile;
        // Create file stream
        std::fstream ifs(tempfile.GetFilename(), std::ios::out);
        // Close stream -> create file
        ifs.close();
        // Open in non-truncate mode
        StorageWriter(tempfile.GetFilename(), /* truncateFile */ false);
        storage_header header;
        TestUtil::ReadData(tempfile.GetFilename(), header, 0);
        ML_ASSERT_EQ(header.count, uint32_t(0));
        ML_ASSERT_EQ(header.version, uint32_t(2));
    }

    // file exists, with header
    {
        AutoDeleteTempFile tempfile;

        {  // Creates Storage writer on empty file i.e. adds header
            StorageWriter writer(tempfile.GetFilename());
        }
        // Opening file with header for writing
        StorageWriter(tempfile.GetFilename(), /* truncateFile */ false);
        storage_header header;
        TestUtil::ReadData(tempfile.GetFilename(), header, 0);
        ML_ASSERT_EQ(header.count, uint32_t(0));
        ML_ASSERT_EQ(header.version, uint32_t(2));
    }

    // File exists with data less than header size
    {
        AutoDeleteTempFile tempfile;
        std::vector<uint8_t> bytes{0x0000, 0x0000, 0x0000, 0x0000};
        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(bytes));
        ML_ASSERT_THROWS_WITH_MESSAGE(StorageWriter(tempfile.GetFilename(), /* truncateFile */ false),
                                      std::runtime_error,
                                      "Incorrect file header, please use truncateFile");
    }

    // File exists with data more than header size
    {
        AutoDeleteTempFile tempfile;
        std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(data));
        ML_ASSERT_THROWS_WITH_MESSAGE(StorageWriter(tempfile.GetFilename(), /* truncateFile */ false),
                                      std::runtime_error,
                                      "Incorrect file header, please use truncateFile");
    }

    return 0;
}

int testStorageWriterTestsAlignment()
{
    AutoDeleteTempFile tempfile;
    {
        const std::vector<float> data{0x0000};
        uint64_t firstOffset = 0, secondOffset = 0;
        {
            StorageWriter writer(tempfile.GetFilename());
            firstOffset = writer.WriteData(Util::MakeSpan(data));
            ML_ASSERT_EQ(firstOffset % DefaultStorageAlignment, uint64_t(0));
            secondOffset = writer.WriteData(Util::MakeSpan(data));
            ML_ASSERT_EQ(secondOffset % DefaultStorageAlignment, uint64_t(0));
        }
        blob_metadata firstMetadata, secondMetadata;
        TestUtil::ReadData<blob_metadata>(tempfile.GetFilename(), firstMetadata, firstOffset);
        TestUtil::ReadData<blob_metadata>(tempfile.GetFilename(), secondMetadata, secondOffset);
        ML_ASSERT_EQ(firstMetadata.offset % DefaultStorageAlignment, uint64_t(0));
        ML_ASSERT_EQ(secondMetadata.offset % DefaultStorageAlignment, uint64_t(0));
        // Second metadata must be aligned i.e.
        // In current test, <first metadata><first blob data>-60 bytes-<second metadata><second blob data>
        ML_ASSERT_EQ(secondOffset - (firstMetadata.offset + firstMetadata.sizeInBytes),
                     DefaultStorageAlignment - firstMetadata.sizeInBytes % DefaultStorageAlignment);
    }

    return 0;
}
