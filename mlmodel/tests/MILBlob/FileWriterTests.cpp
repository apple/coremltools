// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/FileWriter.hpp"
#include "MILBlob/Blob/StorageFormat.hpp"
#include "MILBlob/Util/SpanCast.hpp"
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
template <typename T>
Util::Span<uint8_t> CastAndMakeSpan(T& x)
{
    return Util::Span<uint8_t>(reinterpret_cast<uint8_t*>(&x), sizeof(x));
}
}  // anonymous namespace


int testFileWriterTestsNoAccess()
{
    std::string fileName = "";
    {
        AutoDeleteTempFile tmpDir(AutoDeleteTempFile::FileType::DIR);
        fileName = tmpDir.GetFilename() + "/weight.wt";
    }
    ML_ASSERT_THROWS_WITH_MESSAGE(FileWriter(fileName, /* truncateFile */ true), std::runtime_error, "Unable to open");

    return 0;
}

int testFileWriterTestsWriteToFile()
{
    AutoDeleteTempFile tempfile;
    // Writing to empty file
    {
        uint64_t offset = 0;
        const std::vector<uint8_t> expected{10, 20, 30, 40};
        auto expectedSpan = Util::MakeSpan(expected);
        {
            FileWriter writer(tempfile.GetFilename(), /* truncateFile */ true);
            writer.WriteData(Util::SpanCast<const uint8_t>(expectedSpan), offset);
        }
        std::vector<uint8_t> output(expected.size());
        auto outputSpan = Util::MakeSpan(output);
        TestUtil::ReadBlobFile<uint8_t>(tempfile.GetFilename(), offset, outputSpan);
        ML_ASSERT_SPAN_EQ(expectedSpan, outputSpan);
    }

    // Writing to empty file without truncating
    {
        uint64_t offset = 0;
        const std::vector<uint8_t> expected{10, 20, 30, 40};
        auto expectedSpan = Util::MakeSpan(expected);
        {
            FileWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            writer.WriteData(Util::SpanCast<const uint8_t>(expectedSpan), offset);
        }
        std::vector<uint8_t> output(expected.size());
        auto outputSpan = Util::MakeSpan(output);
        TestUtil::ReadBlobFile<uint8_t>(tempfile.GetFilename(), offset, outputSpan);
        ML_ASSERT_SPAN_EQ(expectedSpan, outputSpan);
    }

    // Writing to existing file without truncating
    {
        uint64_t offset = 64;
        const std::vector<float> expected{1.235f, 5.674f, 642.f, 891.41f, 14124.14f};
        auto expectedSpan = Util::MakeSpan(expected);
        {
            FileWriter writer(tempfile.GetFilename(), /* truncateFile */ false);
            writer.WriteData(Util::SpanCast<const uint8_t>(expectedSpan), offset);
        }
        std::vector<float> output(expected.size());
        auto outputSpan = Util::MakeSpan(output);
        TestUtil::ReadBlobFile<float>(tempfile.GetFilename(), offset, outputSpan);
        ML_ASSERT_SPAN_EQ(expectedSpan, outputSpan);

        // Ensure earlier written data is present as well
        const std::vector<uint8_t> oldExpected{10, 20, 30, 40};
        auto oldExpectedSpan = Util::MakeSpan(oldExpected);
        std::vector<uint8_t> outputOld(oldExpected.size());
        auto outputOldSpan = Util::MakeSpan(outputOld);
        TestUtil::ReadBlobFile<uint8_t>(tempfile.GetFilename(), 0, outputOldSpan);
        ML_ASSERT_SPAN_EQ(oldExpectedSpan, outputOldSpan);
    }

    return 0;
}

int testFileWriterTestsWriteDataWithOffset()
{
    AutoDeleteTempFile tempfile;
    // Writing to empty file
    {
        uint64_t offset = 64;
        std::vector<uint8_t> expected{10, 20, 30, 40};
        auto expectedSpan = Util::MakeSpan(expected);
        {
            FileWriter writer(tempfile.GetFilename(), /* truncateFile */ true);
            writer.WriteData(Util::SpanCast<uint8_t>(expectedSpan), offset);
        }
        std::vector<uint8_t> output(expected.size());
        auto outputSpan = Util::MakeSpan(output);
        TestUtil::ReadBlobFile<uint8_t>(tempfile.GetFilename(), offset, outputSpan);
        ML_ASSERT_SPAN_EQ(expectedSpan, outputSpan);

        // Writer writes 0s if offset is out of current file boundary
        expected = {0, 0, 0, 0};
        TestUtil::ReadBlobFile<uint8_t>(tempfile.GetFilename(), 0, outputSpan);
        ML_ASSERT_SPAN_EQ(expectedSpan, outputSpan);
    }

    return 0;
}

int testFileWriterTestsOffsetNotAligned()
{
    AutoDeleteTempFile tempfile;
    // Writing to empty file
    {
        uint64_t offset = 3;
        std::vector<uint8_t> expected{10, 20, 30, 40};
        auto expectedSpan = Util::MakeSpan(expected);

        FileWriter writer(tempfile.GetFilename(), /* truncateFile */ true);
        ML_ASSERT_THROWS_WITH_MESSAGE(writer.WriteData(Util::SpanCast<uint8_t>(expectedSpan), offset),
                                      std::runtime_error,
                                      "Provided offset not aligned.");
    }

    return 0;
}

int testFileWriterTestsReadData()
{
    AutoDeleteTempFile tempfile;
    // Reading Data from file via Writer
    {
        struct data {
            char a;
            int b;
            float c;
            char d;
        };

        struct data_p1 {
            char a;
            int b;
        };

        struct data_p2 {
            float c;
            char d;
        };

        data data1;
        data1.a = 'b';
        data1.b = 23;
        data1.c = 0.234513f;
        data1.d = 's';

        data_p1 out1;
        data_p2 out2;

        uint64_t offset1 = 0;
        uint64_t offset2 = sizeof(out1);
        {
            TestUtil::WriteBlobFile(tempfile.GetFilename(), CastAndMakeSpan(data1));
        }

        FileWriter writer(tempfile.GetFilename(), /* truncateFile */ false);

        writer.ReadData(offset1, CastAndMakeSpan(out1));
        writer.ReadData(offset2, CastAndMakeSpan(out2));

        // Validate data
        ML_ASSERT_EQ(data1.a, out1.a);
        ML_ASSERT_EQ(data1.b, out1.b);
        ML_ASSERT_EQ(data1.c, out2.c);
        ML_ASSERT_EQ(data1.d, out2.d);

        // Unable to read data i.e. reached end of file
        ML_ASSERT_THROWS_WITH_MESSAGE(writer.ReadData(offset2, CastAndMakeSpan(data1)),
                                      std::runtime_error,
                                      "Unknown error occurred while reading data from the file.");
    }

    return 0;
}
