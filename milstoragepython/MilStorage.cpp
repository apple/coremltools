// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MilStorage.hpp"
#include "MILBlob/SubByteTypes.hpp"
#include "MILBlob/Blob/StorageReader.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Util/SpanCast.hpp"
#include "MILBlob/Util/SubByteConversionUtils.hpp"

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

    template <typename T>
    u_int64_t writeUnsignedSubByteData(MILBlob::Blob::StorageWriter& m_writer,
                                       const py::array_t<uint8_t>& data) {
        // The `data` is stored in uint8 as numpy doesn't support uint1/2/3/4/6 (denoted as uint{x}).
        // First pack those uint{x} data into uint8 Span, and then cast to uint{x} Span and write it.
        auto uint8SpanData = MILBlob::Util::Span<const uint8_t>(data.data(), data.size());
        std::vector<uint8_t> packedValues = MILBlob::PackUInt8Span<T>(uint8SpanData);
        auto packedValuesSpan = MILBlob::Util::Span<const uint8_t>(packedValues.data(), packedValues.size());
        auto uintSubByteSpan = MILBlob::Util::CastToBitSpan<const T>(packedValuesSpan, data.size());
        return m_writer.WriteData<T>(uintSubByteSpan);
    }

}

// These methods are needed in addition to the above template methods
// because pybind does not allow us to expose template methods to
// Python with gcc on Linux.
u_int64_t MilStoragePythonWriter::write_int4_data(const py::array_t<int8_t>& data) {
    // The `data` is stored in int8 because numpy doesn't support int4.
    // First pack those int4 data into uint8 Span, and then cast to int4 Span and write it.
    auto int8SpanData = MILBlob::Util::Span<const int8_t>(data.data(), data.size());
    std::vector<uint8_t> packedValues = MILBlob::PackInt8Span<MILBlob::Int4>(int8SpanData);
    auto packedValuesSpan = MILBlob::Util::Span<const uint8_t>(packedValues.data(), packedValues.size());
    auto int4Span = MILBlob::Util::CastToBitSpan<const MILBlob::Int4>(packedValuesSpan, data.size());
    return m_writer->WriteData<MILBlob::Int4>(int4Span);
}

u_int64_t MilStoragePythonWriter::write_uint1_data(const py::array_t<uint8_t>& data) {
    return writeUnsignedSubByteData<MILBlob::UInt1>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint2_data(const py::array_t<uint8_t>& data) {
    return writeUnsignedSubByteData<MILBlob::UInt2>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint3_data(const py::array_t<uint8_t>& data) {
    return writeUnsignedSubByteData<MILBlob::UInt3>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint4_data(const py::array_t<uint8_t>& data) {
    return writeUnsignedSubByteData<MILBlob::UInt4>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint6_data(const py::array_t<uint8_t>& data) {
    return writeUnsignedSubByteData<MILBlob::UInt6>(*m_writer, data);
}

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

u_int64_t MilStoragePythonWriter::write_int32_data(const py::array_t<int32_t>& data) {
    return writeData<int32_t>(*m_writer, data);
}

u_int64_t MilStoragePythonWriter::write_uint32_data(const py::array_t<uint32_t>& data) {
    return writeData<uint32_t>(*m_writer, data);
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

    template <typename T>
    py::array_t<uint8_t> readUnsignedSubByteData(MILBlob::Blob::StorageReader& m_reader,
                                           uint64_t offset) {
        // First read packed data using MILBlob reader, and restore to uint8 values which represents uint{x}.
        auto uintSubByteSpanData = m_reader.GetDataView<T>(offset);
        MILBlob::Util::Span<const uint8_t> packedValuesSpan = MILBlob::Util::CastFromBitSpan<const T>(uintSubByteSpanData);
        auto unpackedUIntSubByteData = MILBlob::UnPackSubByteVec<T>({packedValuesSpan.begin(), packedValuesSpan.end()}, uintSubByteSpanData.Size());
        return py::array_t<uint8_t>(unpackedUIntSubByteData.size(), reinterpret_cast<uint8_t*>(unpackedUIntSubByteData.data()));
    }
}

// These methods are needed in addition to the above template methods
// because pybind does not allow us to expose template methods to
// Python with gcc on Linux.
py::array_t<int8_t> MilStoragePythonReader::read_int4_data(uint64_t offset) {
    // First read packed data using MILBlob reader, and restore to int8 values which represents int4.
    auto int4SpanData = m_reader->GetDataView<MILBlob::Int4>(offset);
    MILBlob::Util::Span<const uint8_t> packedValuesSpan = MILBlob::Util::CastFromBitSpan<const MILBlob::Int4>(int4SpanData);
    auto unpackedInt4Data = MILBlob::UnPackSubByteVec<MILBlob::Int4>({packedValuesSpan.begin(), packedValuesSpan.end()}, int4SpanData.Size());
    return py::array_t<int8_t>(unpackedInt4Data.size(), reinterpret_cast<int8_t*>(unpackedInt4Data.data()));
}

py::array_t<uint8_t> MilStoragePythonReader::read_uint1_data(uint64_t offset) {
    return readUnsignedSubByteData<MILBlob::UInt1>(*m_reader, offset);
}

py::array_t<uint8_t> MilStoragePythonReader::read_uint2_data(uint64_t offset) {
    return readUnsignedSubByteData<MILBlob::UInt2>(*m_reader, offset);
}

py::array_t<uint8_t> MilStoragePythonReader::read_uint3_data(uint64_t offset) {
    return readUnsignedSubByteData<MILBlob::UInt3>(*m_reader, offset);
}

py::array_t<uint8_t> MilStoragePythonReader::read_uint4_data(uint64_t offset) {
    return readUnsignedSubByteData<MILBlob::UInt4>(*m_reader, offset);
}

py::array_t<uint8_t> MilStoragePythonReader::read_uint6_data(uint64_t offset) {
    return readUnsignedSubByteData<MILBlob::UInt6>(*m_reader, offset);
}

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

py::array_t<int32_t> MilStoragePythonReader::read_int32_data(uint64_t offset) {
    return readData<int32_t>(*m_reader, offset);
}

py::array_t<uint32_t> MilStoragePythonReader::read_uint32_data(uint64_t offset) {
    return readData<uint32_t>(*m_reader, offset);
}

py::array_t<uint16_t> MilStoragePythonReader::read_fp16_data(uint64_t offset) {

    auto fpView = m_reader->GetDataView<MILBlob::Fp16>(offset);
    auto intView = MILBlob::Util::SpanCast<const uint16_t>(fpView);

    return py::array_t<uint16_t> (intView.Size(), intView.Data());
}

py::array_t<float> MilStoragePythonReader::read_float_data(uint64_t offset) {
    return readData<float>(*m_reader, offset);
}
