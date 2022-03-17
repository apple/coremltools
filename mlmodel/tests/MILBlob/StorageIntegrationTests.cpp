// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/StorageReader.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/SpanCast.hpp"
#include "AutoDeleteTempFile.hpp"
#include "framework/TestUtils.hpp"
#include "MLModelTests.hpp"

#include <stdexcept>

using namespace MILBlob;
using namespace MILBlob::Blob;
using MILBlob::TestUtil::AutoDeleteTempFile;


int testStorageIntegrationTestsWriteAndReadValues()
{
    AutoDeleteTempFile tempfile;

    const std::vector<uint8_t> data0 = {0x02, 0x00, 0x40, 0x00, 0x07};
    const std::vector<Fp16> data1 = {Fp16(0x000E), Fp16(0xC0FE), Fp16(0x0810)};
    const std::vector<float> data2 = {0x700000, 0xC0FEE, 0x8FACE, 0x91FADE};

    uint64_t offset0, offset1, offset2;
    {
        StorageWriter writer(tempfile.GetFilename());
        // offset in bytes for reference
        // header: 0
        // offset0: 64
        // metadata0: 64 - 127,
        // data0: 128 - 131
        offset0 = writer.WriteData(Util::MakeSpan(data0));
        // padding: 132 - 191
        // offset1: 192
        // metadata0: 192 - 255,
        // data0: 256 - 261
        offset1 = writer.WriteData(Util::MakeSpan(data1));
        // padding: 262 - 319
        // offset1: 320
        // metadata0: 320 - 384,
        // data0: 384 - 400
        offset2 = writer.WriteData(Util::MakeSpan(data2));
    }

    StorageReader reader(tempfile.GetFilename());
    // Validate data0
    ML_ASSERT_EQ(offset0, uint64_t(64));
    auto out0 = reader.GetDataView<uint8_t>(offset0);
    ML_ASSERT_SPAN_EQ(Util::MakeSpan(data0), out0);

    // Validate data1
    ML_ASSERT_EQ(offset1, uint64_t(192));
    auto out1 = reader.GetDataView<Fp16>(offset1);
    ML_ASSERT_SPAN_EQ(Util::MakeSpan(data1), out1);

    // Validate data2
    ML_ASSERT_EQ(offset2, uint64_t(320));
    auto out2 = reader.GetDataView<float>(offset2);
    ML_ASSERT_SPAN_EQ(Util::MakeSpan(data2), out2);

    return 0;
}

int testStorageIntegrationTestsReadDataWithIncorrectType()
{
    AutoDeleteTempFile tempfile;

    const std::vector<uint8_t> data = {0x02, 0x00, 0x40, 0x00, 0x07};
    uint64_t offset;
    {
        StorageWriter writer(tempfile.GetFilename());
        offset = writer.WriteData(Util::MakeSpan(data));
    }

    StorageReader reader(tempfile.GetFilename());
    {
        ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<float>(offset),
                                      std::runtime_error,
                                      "Metadata data type does not match requested type.");
    }

    return 0;
}

int testStorageIntegrationTestsReadDataWithIncorrectOffset()
{
    AutoDeleteTempFile tempfile;

    const std::vector<uint8_t> data = {0x02, 0x00, 0x40, 0x00, 0x07};
    {
        StorageWriter writer(tempfile.GetFilename());
        writer.WriteData(Util::MakeSpan(data));
    }

    StorageReader reader(tempfile.GetFilename());
    {
        ML_ASSERT_THROWS_WITH_MESSAGE(reader.GetDataView<uint8_t>(0),
                                      std::runtime_error,
                                      "Invalid sentinel in blob_metadata.");
    }

    return 0;
}

