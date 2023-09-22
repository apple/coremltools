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
                        const py::array_t<T>& data) {
        auto fpSpan = MILBlob::Util::Span<const T>(data.data(), data.size());
        return m_writer.WriteData(fpSpan);
    }

}

// These methods are needed in addition to the above template methods
// because pybind does not allow us to expose template methods to
// Python with gcc on Linux.
u_int64_t MilStoragePythonWriter::write_int8_data(const py::array_t<int8_t>& data) {
    return writeData<int8_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint8_data(const py::array_t<uint8_t>& data) {
    return writeData<uint8_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_int16_data(const py::array_t<int16_t>& data) {
    return writeData<int16_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint16_data(const py::array_t<uint16_t>& data) {
    return writeData<uint16_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_fp16_data(const py::array_t<uint16_t>& data){

    auto intSpan = MILBlob::Util::Span<const uint16_t>(data.data(), data.size());
    auto fpSpan = MILBlob::Util::SpanCast<const MILBlob::Fp16>(intSpan);

    return m_writer->WriteData(fpSpan);
}

u_int64_t MilStoragePythonWriter::write_float_data(const py::array_t<float>& data){
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
    py::array_t<T> readData(MILBlob::Blob::StorageReader& m_reader,
                                  uint64_t offset) {
        auto spanData = m_reader.GetDataView<T>(offset);
        return py::array_t<T>(spanData.Size(), spanData.Data());
    }
}

// These methods are needed in addition to the above template methods
// because pybind does not allow us to expose template methods to
// Python with gcc on Linux.
py::array_t<int8_t> MilStoragePythonReader::read_int8_data(uint64_t offset) {
    return readData<int8_t>(*m_reader, offset);
}

py::array_t<uint8_t> MilStoragePythonReader::read_uint8_data(uint64_t offset) {
    return readData<uint8_t>(*m_reader, offset);
}

py::array_t<int16_t> MilStoragePythonReader::read_int16_data(uint64_t offset) {
    return readData<int16_t>(*m_reader, offset);
}

py::array_t<uint16_t> MilStoragePythonReader::read_uint16_data(uint64_t offset) {
    return readData<uint16_t>(*m_reader, offset);
}

py::array_t<uint16_t> MilStoragePythonReader::read_fp16_data(uint64_t offset) {

    auto fpView = m_reader->GetDataView<MILBlob::Fp16>(offset);
    auto intView = MILBlob::Util::SpanCast<const uint16_t>(fpView);

    return py::array_t<uint16_t> (intView.Size(), intView.Data());
}

py::array_t<float> MilStoragePythonReader::read_float_data(uint64_t offset) {
    return readData<float>(*m_reader, offset);
}