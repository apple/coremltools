// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <string>

namespace MILBlob {
namespace TestUtil {

class AutoDeleteTempFile {
public:
    enum FileType {
        FILE = 0,
        DIR = 1
    };

    /**
     * Creates an empty file in /tmp.
     */
    AutoDeleteTempFile();

    /**
     * Creates an empty file or directory in /tmp.
     * @param type  Whether to create a directory or a regular file.
     */
    AutoDeleteTempFile(FileType type);

    ~AutoDeleteTempFile();

    AutoDeleteTempFile(const AutoDeleteTempFile&) = delete;
    AutoDeleteTempFile& operator=(const AutoDeleteTempFile&) = delete;

    AutoDeleteTempFile(AutoDeleteTempFile&&) = default;
    AutoDeleteTempFile& operator=(AutoDeleteTempFile&&) = default;

    const std::string& GetFilename() const;

private:
    std::string m_filename;
    int m_fileDescriptor;
};

}  // namespace TestUtil
}  // namespace MILBlob
