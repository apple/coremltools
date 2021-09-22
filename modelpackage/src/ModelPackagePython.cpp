//
//  ModelPackagePython.hpp
//  CoreML Model Package
//
//  Copyright © 2021 Apple Inc. All rights reserved.
//

#include "ModelPackage.hpp"

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

PYBIND11_PLUGIN(libmodelpackage) {
    py::module m("libmodelpackage", "Library to create, access and edit model packages");
    
    py::class_<MPL::ModelPackageItemInfo>(m, "ModelPackageItemInfo")
        .def("identifier", &MPL::ModelPackageItemInfo::identifier)
        .def("path", &MPL::ModelPackageItemInfo::path)
        .def("name", &MPL::ModelPackageItemInfo::name)
        .def("author", &MPL::ModelPackageItemInfo::author)
        .def("description", &MPL::ModelPackageItemInfo::description);
    
    py::class_<MPL::ModelPackage>(m, "ModelPackage")
        .def(py::init<const std::string&>())
        .def("path", &MPL::ModelPackage::path)
        .def("setRootModel", &MPL::ModelPackage::setRootModel)
        .def("replaceRootModel", &MPL::ModelPackage::replaceRootModel)
        .def("addItem", &MPL::ModelPackage::addItem)
        .def("getRootModel", &MPL::ModelPackage::getRootModel)
        .def("isValid", &MPL::ModelPackage::isValid);
    
    return m.ptr();
}

#pragma clang diagnostic pop
