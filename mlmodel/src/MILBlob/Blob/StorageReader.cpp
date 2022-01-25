// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/MMapFileReader.hpp"
#include "MILBlob/Blob/MMapFileReaderFactory.hpp"
#include "MILBlob/Blob/StorageFormat.hpp"
#include "MILBlob/Blob/StorageReader.hpp"
#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/SpanCast.hpp"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

using namespace MILBlob;
using namespace MILBlob::Blob;

class StorageReader::Impl final {
public:
    Impl(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl& operator=(Impl&&) = delete;

    Impl(std::string filename) : m_filePath(std::move(filename)) {}
    ~Impl() = default;

    const std::string& GetFilename() const
    {
        return m_filePath;
    }

    blob_metadata GetMetadata(uint64_t offset) const
    {
        EnsureLoaded();

        blob_metadata metadata = m_reader->ReadStruct<blob_metadata>(offset);

        // validate sentinel
        MILVerifyIsTrue(metadata.sentinel == BlobMetadataSentinel,
                        std::runtime_error,
                        "Invalid sentinel in blob_metadata.");
        return metadata;
    }

    Util::Span<const uint8_t> GetRawDataView(uint64_t offset) const
    {
        auto metadata = GetMetadata(offset);

        return m_reader->ReadData(metadata.offset, metadata.sizeInBytes);
    }

    template <typename T>
    Util::Span<const T> GetDataView(uint64_t offset) const;

    uint64_t GetDataOffset(uint64_t offset) const
    {
        auto metadata = GetMetadata(offset);
        return metadata.offset;
    }

private:
    void EnsureLoaded() const
    {
        auto load = [this]() {
            auto reader = MakeMMapFileReader(m_filePath);
            const storage_header& header = reader->ReadStruct<storage_header>(0);
            MILVerifyIsTrue(header.version == 2, std::runtime_error, "Storage Reader expects file format version 2.");

            // once we're good with the structure of the file, then set class state
            m_reader = std::move(reader);
        };

        std::call_once(m_loadedFlag, [&load]() { load(); });
    }

    const std::string m_filePath;

    mutable std::once_flag m_loadedFlag;
    mutable std::unique_ptr<const MMapFileReader> m_reader;
};

template <typename T>
Util::Span<const T> StorageReader::Impl::GetDataView(uint64_t offset) const
{
    auto metadata = GetMetadata(offset);

    MILVerifyIsTrue(metadata.mil_dtype == BlobDataTypeTraits<T>::DataType,
                    std::runtime_error,
                    "Metadata data type does not match requested type.");
    return Util::SpanCast<const T>(m_reader->ReadData(metadata.offset, metadata.sizeInBytes));
}

// --------------------------------------------------------------------------------------

StorageReader::~StorageReader() = default;

StorageReader::StorageReader(std::string filename) : m_impl(std::make_unique<Impl>(std::move(filename))) {}

const std::string& StorageReader::GetFilename() const
{
    return m_impl->GetFilename();
}

template <>
Util::Span<const uint8_t> StorageReader::GetDataView<uint8_t>(uint64_t offset) const
{
    return m_impl->GetDataView<uint8_t>(offset);
}

template <>
Util::Span<const Fp16> StorageReader::GetDataView<Fp16>(uint64_t offset) const
{
    return m_impl->GetDataView<Fp16>(offset);
}

template <>
Util::Span<const float> StorageReader::GetDataView<float>(uint64_t offset) const
{
    return m_impl->GetDataView<float>(offset);
}

Util::Span<const uint8_t> StorageReader::GetRawDataView(uint64_t offset) const
{
    return m_impl->GetRawDataView(offset);
}

uint64_t StorageReader::GetDataOffset(uint64_t offset) const
{
    return m_impl->GetDataOffset(offset);
}
