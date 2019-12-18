//
//  ProgramModelUtils.hpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 12/12/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "Format.hpp"

namespace ProgramModelUtils {
using namespace ::CoreML::Specification;

using ArgBindings = std::vector<std::pair<std::string, std::string>>;
using Attributes = std::vector<std::pair<std::string, V5::Value>>;
using NameAndTypeVec = std::vector<std::pair<std::string, V5::ValueType>>;
using NameVec = std::vector<std::string>;
using OpVec = std::vector<V5::Operation>;
using TypeVec = std::vector<V5::ValueType>;

V5::ValueType MakeScalarValueType(V5::ScalarType scalarType);
V5::Value MakeBoolValue(bool b);
V5::Value MakeFloatValue(float f);
V5::Dimension MakeDim(int64_t size);
V5::ValueType MakeTensorValueType(V5::ScalarType scalarType, const std::vector<V5::Dimension>& dims);
V5::Block MakeBlock(const ArgBindings& bindings, const NameVec& outputs, const OpVec& ops);
V5::Function MakeFunction(const NameAndTypeVec& params, const TypeVec& outputs, const V5::Block& block);
V5::Operation MakeOp(const std::string& opName, const std::string& opType, const ArgBindings& bindings,
                     const NameAndTypeVec& outputs, const Attributes& attributes);
}
