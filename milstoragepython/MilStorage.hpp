// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace MILBlob {
namespace Blob {
class StorageReader;
class StorageWriter;
}
}

namespace CoreML {
    namespace MilStoragePython {

        class MilStoragePythonWriter {
        public:
            MilStoragePythonWriter(const MilStoragePythonWriter&) = delete;
            MilStoragePythonWriter(MilStoragePythonWriter&&) = delete;
            MilStoragePythonWriter& operator=(const MilStoragePythonWriter&) = delete;
            MilStoragePythonWriter& operator=(MilStoragePythonWriter&&) = delete;

            MilStoragePythonWriter(const std::string& filePath, bool truncateFile);
            ~MilStoragePythonWriter();

            u_int64_t write_int4_data(const py::array_t<int8_t>& data);
            u_int64_t write_uint1_data(const py::array_t<uint8_t>& data);
            u_int64_t write_uint2_data(const py::array_t<uint8_t>& data);
            u_int64_t write_uint3_data(const py::array_t<uint8_t>& data);
            u_int64_t write_uint4_data(const py::array_t<uint8_t>& data);
            u_int64_t write_uint6_data(const py::array_t<uint8_t>& data);
            u_int64_t write_int8_data(const py::array_t<int8_t>& data);
            u_int64_t write_uint8_data(const py::array_t<uint8_t>& data);
            u_int64_t write_int16_data(const py::array_t<int16_t>& data);
            u_int64_t write_uint16_data(const py::array_t<uint16_t>& data);
            u_int64_t write_int32_data(const py::array_t<int32_t>& data);
            u_int64_t write_uint32_data(const py::array_t<uint32_t>& data);
            u_int64_t write_fp16_data(const py::array_t<uint16_t>& data);
            u_int64_t write_float_data(const py::array_t<float>& data);

        private:
            std::unique_ptr<MILBlob::Blob::StorageWriter> m_writer;
        };

        class MilStoragePythonReader {
        public:
            MilStoragePythonReader(const MilStoragePythonReader&) = delete;
            MilStoragePythonReader(MilStoragePythonReader&&) = delete;
            MilStoragePythonReader& operator=(const MilStoragePythonReader&) = delete;
            MilStoragePythonReader& operator=(MilStoragePythonReader&&) = delete;

            MilStoragePythonReader(std::string filePath);
            ~MilStoragePythonReader();

            py::array_t<int8_t> read_int4_data(uint64_t offset);
            py::array_t<uint8_t> read_uint1_data(uint64_t offset);
            py::array_t<uint8_t> read_uint2_data(uint64_t offset);
            py::array_t<uint8_t> read_uint3_data(uint64_t offset);
            py::array_t<uint8_t> read_uint4_data(uint64_t offset);
            py::array_t<uint8_t> read_uint6_data(uint64_t offset);
            py::array_t<int8_t> read_int8_data(uint64_t offset);
            py::array_t<uint8_t> read_uint8_data(uint64_t offset);
            py::array_t<int16_t> read_int16_data(uint64_t offset);
            py::array_t<uint16_t> read_uint16_data(uint64_t offset);
            py::array_t<int32_t> read_int32_data(uint64_t offset);
            py::array_t<uint32_t> read_uint32_data(uint64_t offset);
            py::array_t<uint16_t> read_fp16_data(uint64_t offset);
            py::array_t<float> read_float_data(uint64_t offset);


        private:
            std::unique_ptr<MILBlob::Blob::StorageReader> m_reader;
        };
    }
}
