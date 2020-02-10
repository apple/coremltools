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
using BlockVec = std::vector<V5::Block>;
using NameAndTypeVec = std::vector<std::pair<std::string, V5::ValueType>>;
using NameVec = std::vector<std::string>;
using OpVec = std::vector<V5::Operation>;
using TypeVec = std::vector<V5::ValueType>;

/** Make an NNv2 scalar type. */
V5::ValueType MakeScalarValueType(V5::ScalarType scalarType);

/** Make an NNv2 tensor type. */
V5::ValueType MakeTensorValueType(V5::ScalarType scalarType, const std::vector<V5::Dimension>& dims);

/** Make an NNv2 boolean value. */
V5::Value MakeBoolValue(bool b);

/** Make an NNv2 float32 value. */
V5::Value MakeFloatValue(float f);

/** Make an NNv2 int32 value. */
V5::Value MakeIntValue(int i);

/** Make an NNv2 string value. */
V5::Value MakeStringValue(std::string s);

/** Make an NNv2 file value. */
V5::Value MakeFileValue(const std::string& file_name, uint64_t offset, const V5::ValueType& valueType);

/** Make an NNv2 tensor of float32 value. */
V5::Value MakeFloatTensorValue(const std::vector<V5::Dimension>& dims, const std::vector<float>& fs);

/** Make an NNv2 tensor of int32 value. */
V5::Value MakeIntTensorValue(const std::vector<V5::Dimension>& dims, const std::vector<int>& is);

/** Make an NNv2 constant dimension. */
V5::Dimension MakeDim(int64_t size);

/** Make an NNv2 block. */
V5::Block MakeBlock(const ArgBindings& bindings, const NameVec& outputs, const OpVec& ops);

/** Make an NNv2 function. */
V5::Function MakeFunction(const NameAndTypeVec& params, const TypeVec& outputs, const V5::Block& block);

/** Make an NNv2 operation. */
V5::Operation MakeOp(const std::string& opName, const std::string& opType, const ArgBindings& bindings,
                     const NameAndTypeVec& outputs, const Attributes& attributes);

/** Make an NNv2 operation. */
V5::Operation MakeOp(const std::string& opName, const std::string& opType, const ArgBindings& bindings,
    const NameAndTypeVec& outputs, const Attributes& attributes, const BlockVec& blocks);

/** Make an NNv2 model feature (input) description. */
FeatureDescription MakeFeatureDescription(const std::string& name, const V5::ValueType& type);

/** Make an NNv2 model image feature description. */
FeatureDescription MakeImageFeatureDescription(const std::string& name, ImageFeatureType_ColorSpace colorSpace,
                                               int64_t height, int64_t width);

/** Make an NNv2 model with an empty main(). */
Model EmptyProgram();

/**
 Create an NNv2 model from a main().

 Assuming the main function is well-formed, this infers model inputs and outputs.
 */
Model ProgramWithMain(const V5::Function& main);

} // namespace ProgramModelUtils
