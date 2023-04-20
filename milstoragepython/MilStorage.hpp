// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <memory>
#include <string>
#include <vector>


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

            u_int64_t write_int8_data(const std::vector<const int8_t>& data);
            u_int64_t write_uint8_data(const std::vector<const uint8_t>& data);
            u_int64_t write_fp16_data(const std::vector<const uint16_t>& data);
            u_int64_t write_float_data(const std::vector<const float>& data);

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

            const std::vector<int8_t> read_int8_data(uint64_t offset);
            const std::vector<uint8_t> read_uint8_data(uint64_t offset);
            const std::vector<uint16_t> read_fp16_data(uint64_t offset);
            const std::vector<float> read_float_data(uint64_t offset);

        private:
            std::unique_ptr<MILBlob::Blob::StorageReader> m_reader;
        };
    }
}
