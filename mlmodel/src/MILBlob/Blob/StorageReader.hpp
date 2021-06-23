// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/Span.hpp"
#include <memory>
#include <string>

namespace MILBlob {
namespace Blob {

/**
 * StorageReader encapsulates memory-mapped reading of the Storage Blob Format.
 *
 * Memory-mapping is performed laziliy on first access to the underlying data.
 *
 * This file format supports the following types:
 * - uint8_t
 * - Fp16
 * - float
 */
class StorageReader final {
public:
    StorageReader() = delete;
    StorageReader(const StorageReader&) = delete;
    StorageReader(StorageReader&&) = delete;
    StorageReader& operator=(const StorageReader&) = delete;
    StorageReader& operator=(StorageReader&&) = delete;

    StorageReader(std::string filename);
    ~StorageReader();

    const std::string& GetFilename() const;

    /**
     * Returns a Span view into the underlying memory-mapped storage. The
     * file will be mapped into memory on first access. This is valid for the
     * supported types noted above.
     * @throws std::range_error if offset is not valid.
     */
    template <typename T>
    Util::Span<const T> GetDataView(uint64_t offset) const;

    /**
     * Returns an uint8_t Span view into the underlying memory-mapped storage. The
     * file will be mapped into memory on first access. This is valid for the
     * supported types noted above.
     * @throws std::range_error if offset is not valid.
     */
    Util::Span<const uint8_t> GetRawDataView(uint64_t offset) const;

    /**
     * Returns file offset of data from given metadata offset
     * @throws std::range_error if offset is not valid.
     */
    uint64_t GetDataOffset(uint64_t offset) const;

private:
    class Impl;
    const std::unique_ptr<Impl> m_impl;
};

template <>
Util::Span<const uint8_t> StorageReader::GetDataView<uint8_t>(uint64_t) const;
template <>
Util::Span<const Fp16> StorageReader::GetDataView<Fp16>(uint64_t) const;
template <>
Util::Span<const float> StorageReader::GetDataView<float>(uint64_t) const;

}  // namespace Blob
}  // namespace MILBlob
