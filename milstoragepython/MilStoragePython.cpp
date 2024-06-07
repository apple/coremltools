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

    // As we have both pybind for the same MilStoragePythonWriter class, we need to set module_local
    // to avoid conflicts between coremltools and coremltools-internal.
    py::class_<MilStoragePythonWriter> blobStorageWriter(m, "_BlobStorageWriter", py::module_local());
    blobStorageWriter.def(py::init<const std::string&, bool>(), py::arg("file_name"), py::arg("truncate_file") = true)
      .def("write_int4_data", &MilStoragePythonWriter::write_int4_data)
      .def("write_uint1_data", &MilStoragePythonWriter::write_uint1_data)
      .def("write_uint2_data", &MilStoragePythonWriter::write_uint2_data)
      .def("write_uint3_data", &MilStoragePythonWriter::write_uint3_data)
      .def("write_uint4_data", &MilStoragePythonWriter::write_uint4_data)
      .def("write_uint6_data", &MilStoragePythonWriter::write_uint6_data)
      .def("write_int8_data", &MilStoragePythonWriter::write_int8_data)
      .def("write_uint8_data", &MilStoragePythonWriter::write_uint8_data)
      .def("write_int16_data", &MilStoragePythonWriter::write_int16_data)
      .def("write_uint16_data", &MilStoragePythonWriter::write_uint16_data)
      .def("write_int32_data", &MilStoragePythonWriter::write_int32_data)
      .def("write_uint32_data", &MilStoragePythonWriter::write_uint32_data)
      .def("write_fp16_data", &MilStoragePythonWriter::write_fp16_data)
      .def("write_float_data", &MilStoragePythonWriter::write_float_data);

    py::class_<MilStoragePythonReader> blobStorageReader(m, "_BlobStorageReader", py::module_local());
    blobStorageReader.def(py::init<std::string>())
      .def("read_int4_data", &MilStoragePythonReader::read_int4_data)
      .def("read_uint1_data", &MilStoragePythonReader::read_uint1_data)
      .def("read_uint2_data", &MilStoragePythonReader::read_uint2_data)
      .def("read_uint3_data", &MilStoragePythonReader::read_uint3_data)
      .def("read_uint4_data", &MilStoragePythonReader::read_uint4_data)
      .def("read_uint6_data", &MilStoragePythonReader::read_uint6_data)
      .def("read_int8_data", &MilStoragePythonReader::read_int8_data)
      .def("read_uint8_data", &MilStoragePythonReader::read_uint8_data)
      .def("read_int16_data", &MilStoragePythonReader::read_int16_data)
      .def("read_uint16_data", &MilStoragePythonReader::read_uint16_data)
      .def("read_int32_data", &MilStoragePythonReader::read_int32_data)
      .def("read_uint32_data", &MilStoragePythonReader::read_uint32_data)
      .def("read_fp16_data", &MilStoragePythonReader::read_fp16_data)
      .def("read_float_data", &MilStoragePythonReader::read_float_data);

    return m.ptr();
}

#pragma clang diagnostic pop
