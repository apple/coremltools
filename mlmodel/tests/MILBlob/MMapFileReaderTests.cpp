// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/MMapFileReader.hpp"
#include "AutoDeleteTempFile.hpp"
#include "BlobUtils.hpp"
#include "framework/TestUtils.hpp"
#include "MLModelTests.hpp"

#include <stdexcept>
#include <string_view>

using namespace MILBlob;
using namespace MILBlob::Blob;
using MILBlob::TestUtil::AutoDeleteTempFile;


int testMMapFileReaderTestsFileErrorNotFound()
{
    AutoDeleteTempFile tempfile;
    std::remove(tempfile.GetFilename().c_str());
    ML_ASSERT_THROWS_WITH_MESSAGE(MMapFileReader(tempfile.GetFilename()), std::runtime_error, "Could not open");

    return 0;
}

int testMMapFileReaderTestsFileErrorEmpty()
{
    AutoDeleteTempFile tempfile;
    ML_ASSERT_THROWS_WITH_MESSAGE(MMapFileReader(tempfile.GetFilename()), std::runtime_error, "Unable to mmap");

    return 0;
}

int testMMapFileReaderTestsReadData()
{
    AutoDeleteTempFile tempfile;

    std::string_view fileData("salut! ca va?");
    TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::Span<const char>(fileData.data(), fileData.size()));

    MMapFileReader reader(tempfile.GetFilename());
    ML_ASSERT_EQ(reader.GetLength(), uint64_t(13));
    ML_ASSERT_NOT(reader.IsEncrypted());

    ML_ASSERT_THROWS(reader.ReadData(0, 14), std::range_error);
    ML_ASSERT_THROWS(reader.ReadData(13, 1), std::range_error);

    {
        std::string_view expected("salut");
        Util::Span<const char> expectedSpan(expected.data(), expected.size());

        auto span = reader.ReadData(0, 5);
        ML_ASSERT_SPAN_EQ(span, expected);
    }

    {
        std::string_view expected("ca va");
        Util::Span<const char> expectedSpan(expected.data(), expected.size());

        auto span = reader.ReadData(7, 5);
        ML_ASSERT_SPAN_EQ(span, expected);
    }

    {
        Util::Span<const char> expectedSpan(fileData.data(), fileData.size());

        auto span = reader.ReadData(0, reader.GetLength());
        ML_ASSERT_SPAN_EQ(span, expectedSpan);
    }

    return 0;
}

int testMMapFileReaderTestsReadStruct()
{
    AutoDeleteTempFile tempfile;

    {
        std::array<uint16_t, 3> values{10, 20, 30};
        TestUtil::WriteBlobFile(tempfile.GetFilename(), Util::MakeSpan(values));
    }

    MMapFileReader reader(tempfile.GetFilename());
    ML_ASSERT_EQ(reader.GetLength(), uint64_t(6));
    ML_ASSERT_NOT(reader.IsEncrypted());

    struct Int16Data {
        uint16_t a;
        uint16_t b;
    };

    struct Int64Data {
        uint64_t a;
        uint64_t b;
    };

    // test invalid offsets/lengths
    ML_ASSERT_THROWS(reader.ReadStruct<Int16Data>(6), std::range_error);
    ML_ASSERT_THROWS(reader.ReadStruct<Int64Data>(0), std::range_error);

    {
        const Int16Data& data = reader.ReadStruct<Int16Data>(2);
        ML_ASSERT_EQ(data.a, 20);
        ML_ASSERT_EQ(data.b, 30);
    }

    return 0;
}
