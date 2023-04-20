// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MilStorage.hpp"
#include "MILBlob/Blob/StorageReader.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Util/SpanCast.hpp"

#include <memory>

using namespace CoreML::MilStoragePython;


/*
 *
 * MilStoragePythonWriter
 *
 */

MilStoragePythonWriter::~MilStoragePythonWriter() = default;

MilStoragePythonWriter::MilStoragePythonWriter(const std::string& filePath, bool truncateFile)
    : m_writer(std::make_unique<MILBlob::Blob::StorageWriter>(filePath, truncateFile))
{}


namespace {
    template <typename T>
    u_int64_t writeData(MILBlob::Blob::StorageWriter& m_writer,
                        const std::vector<const T>& data) {
        return m_writer.WriteData(MILBlob::Util::MakeSpan(data));
    }

    template <>
    u_int64_t writeData<uint16_t>(MILBlob::Blob::StorageWriter& m_writer,
                                  const std::vector<const uint16_t>& data) {
        auto intSpan = MILBlob::Util::MakeSpan(data);
        auto fpSpan = MILBlob::Util::SpanCast<const MILBlob::Fp16>(intSpan);
        return m_writer.WriteData(fpSpan);
  }
}

// These methods are needed in addition to the above template methods
// because pybind does not allow us to expose template methods to
// Python with gcc on Linux.
u_int64_t MilStoragePythonWriter::write_int8_data(const std::vector<const int8_t>& data) {
    return writeData<int8_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint8_data(const std::vector<const uint8_t>& data) {
    return writeData<uint8_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_fp16_data(const std::vector<const uint16_t>& data) {
    return writeData<uint16_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_float_data(const std::vector<const float>& data) {
    return writeData<float>(*m_writer, data);
}


/*
 *
 * MilStoragePythonReader
 *
 */

MilStoragePythonReader::~MilStoragePythonReader() = default;

MilStoragePythonReader::MilStoragePythonReader(std::string filePath)
    : m_reader(std::make_unique<MILBlob::Blob::StorageReader>(filePath))
{}

namespace {
    template <typename T>
    const std::vector<T> readData(MILBlob::Blob::StorageReader& m_reader,
                                  uint64_t offset) {
        auto view = m_reader.GetDataView<T>(offset);
        return std::vector<T>(view.begin(), view.end());
    }

    template <>
    const std::vector<uint16_t> readData<uint16_t>(MILBlob::Blob::StorageReader& m_reader,
                                                   uint64_t offset) {
        auto fpView = m_reader.GetDataView<MILBlob::Fp16>(offset);
        auto intView = MILBlob::Util::SpanCast<const uint16_t>(fpView);
        return std::vector<uint16_t>(intView.begin(), intView.end());
  }
}

// These methods are needed in addition to the above template methods
// because pybind does not allow us to expose template methods to
// Python with gcc on Linux.
const std::vector<int8_t> MilStoragePythonReader::read_int8_data(uint64_t offset) {
    return readData<int8_t>(*m_reader, offset);
}

const std::vector<uint8_t> MilStoragePythonReader::read_uint8_data(uint64_t offset) {
    return readData<uint8_t>(*m_reader, offset);
}

const std::vector<uint16_t> MilStoragePythonReader::read_fp16_data(uint64_t offset) {
    return readData<uint16_t>(*m_reader, offset);
}

const std::vector<float> MilStoragePythonReader::read_float_data(uint64_t offset) {
    return readData<float>(*m_reader, offset);
}
