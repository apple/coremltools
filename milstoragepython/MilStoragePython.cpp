// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MilStorage.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace py = pybind11;

using namespace CoreML::MilStoragePython;


/*
 *
 * bindings
 *
 */

PYBIND11_PLUGIN(libmilstoragepython) {
    py::module m("libmilstoragepython", "Library to create, access and edit CoreML blob files.");

    py::class_<MilStoragePythonWriter> blobStorageWriter(m, "_BlobStorageWriter", py::module_local());
    blobStorageWriter.def(py::init<const std::string&, bool>(), py::arg("file_name"), py::arg("truncate_file") = true)
      .def("write_int8_data", &MilStoragePythonWriter::write_int8_data)
      .def("write_uint8_data", &MilStoragePythonWriter::write_uint8_data)
      .def("write_int16_data", &MilStoragePythonWriter::write_int16_data)
      .def("write_uint16_data", &MilStoragePythonWriter::write_uint16_data)
      .def("write_fp16_data", &MilStoragePythonWriter::write_fp16_data)
      .def("write_float_data", &MilStoragePythonWriter::write_float_data);

    py::class_<MilStoragePythonReader> blobStorageReader(m, "_BlobStorageReader", py::module_local());
    blobStorageReader.def(py::init<std::string>())
      .def("read_int8_data", &MilStoragePythonReader::read_int8_data)
      .def("read_uint8_data", &MilStoragePythonReader::read_uint8_data)
      .def("read_int16_data", &MilStoragePythonReader::read_int16_data)
      .def("read_uint16_data", &MilStoragePythonReader::read_uint16_data)
      .def("read_fp16_data", &MilStoragePythonReader::read_fp16_data)
      .def("read_float_data", &MilStoragePythonReader::read_float_data);

    return m.ptr();
}

#pragma clang diagnostic pop
