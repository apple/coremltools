// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "AutoDeleteTempFile.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fts.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace MILBlob::TestUtil;

AutoDeleteTempFile::AutoDeleteTempFile() : AutoDeleteTempFile(FILE) {}

AutoDeleteTempFile::AutoDeleteTempFile(FileType type)
    : m_filename("/tmp/com.apple.mil.test_file_temp.XXXXXX")
    , m_fileDescriptor(-1)
{
    switch (type) {
        case FILE:
        {
            m_fileDescriptor = mkstemp(m_filename.data());
            if (m_fileDescriptor == -1) {
                throw std::runtime_error("unable to generate temporary filename");
            }
            break;
        }
        case DIR:
        {
            if (mkdtemp(m_filename.data()) == nullptr) {
                throw std::runtime_error("unable to generate temporary directory");
            }
            chmod(m_filename.data(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
            break;
        }
        default:
            throw std::invalid_argument("Invalid requested file type");
    }
}

namespace {
void DeleteTree(const std::string& path)
{
    // fts_open wants a char* const, not a const char*; therefore, we need
    // a mutable pointer to path. fts promises not to modify it.
    std::vector<char> pathchars(path.begin(), path.end());
    pathchars.push_back(0);

    std::array<char*, 2> paths{pathchars.data(), nullptr};

    FTS* tree = fts_open(paths.data(), FTS_PHYSICAL | FTS_NOCHDIR, nullptr);
    if (!tree) {
        return;
    }

    while (FTSENT* f = fts_read(tree)) {
        switch (f->fts_info) {
            case FTS_D:
                // skip pre-order directory traversal because it's potentially
                // not empty yet
                break;

            default:
                // attempt to delete anything else and ignore errors
                remove(f->fts_accpath);
                break;
        }
    }

    fts_close(tree);
}

}  // namespace

AutoDeleteTempFile::~AutoDeleteTempFile()
{
    if (!m_filename.empty()) {
        DeleteTree(m_filename);

        if (m_fileDescriptor != -1) {
            close(m_fileDescriptor);
        }
    }
}

const std::string& AutoDeleteTempFile::GetFilename() const
{
    return m_filename;
}
