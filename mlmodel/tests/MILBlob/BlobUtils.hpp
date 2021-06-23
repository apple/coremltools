// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Util/Span.hpp"
#include "AutoDeleteTempFile.hpp"

#include <fstream>

namespace MILBlob {
namespace TestUtil {

/**
 * AppendStructToBlobFile: Appends given struct to provided file
 */
template <typename T>
void AppendStructToBlobFile(const std::string& filePath, T* data)
{
    std::ofstream ofs(filePath.c_str(), std::ios::binary | std::ios::app);
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(T)));
}

/**
 * WriteBlobFile: Writes given span to the file
 */
template <typename T, size_t Extent>
void WriteBlobFile(const std::string& filename, Util::Span<T, Extent> data)
{
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(data.Data()), static_cast<std::streamsize>(sizeof(T) * data.Size()));
}

/**
 * Creates an automatically removed temp file and populates its contents with a
 * storage format file with 3 records:
 * offset=40 corresponds to uint8_t data { 0x02, 0x00, 0x40, 0x00, 0x07, 0xC0, 0xD0, 0x0C }
 * offset=104 corresponds to Fp16 data {Fp16(0x000E), Fp16(0xC0FE), Fp16(0x0810), Fp16(0x0000)}
 * offset=168 corresponds to float data {{0x700000, 0xC0FEE, 0x8FACE, 0x91FADE}}
 */
AutoDeleteTempFile MakeStorageTempFileWith3Records();

template <typename T>
void ReadBlobFile(const std::string& filename, uint64_t offset, Util::Span<T> data)
{
    std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::in);
    ifs.seekg(static_cast<std::streamoff>(offset));
    ifs.read(reinterpret_cast<char*>(data.Data()), static_cast<std::streamsize>(sizeof(T) * data.Size()));
}

/**
 * ReadData: Reads data of type T from given filePath at given offset
 */
template <typename T>
void ReadData(const std::string& filePath, T& data, uint64_t offset)
{
    auto dataSpan = Util::Span<uint8_t>(reinterpret_cast<uint8_t*>(&data), sizeof(T));
    ReadBlobFile(filePath, offset, dataSpan);
}

}  // namespace TestUtil
}  // namespace MILBlob
