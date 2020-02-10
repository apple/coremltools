//
//  ProgramModelUtils.cpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 12/12/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ProgramModelUtils.hpp"

namespace ProgramModelUtils {

V5::ValueType MakeScalarValueType(V5::ScalarType scalarType)
{
    V5::ValueType valueType;
    valueType.set_scalartype(scalarType);
    return valueType;
}

V5::ValueType MakeTensorValueType(V5::ScalarType scalarType, const std::vector<V5::Dimension>& dims)
{
    V5::ValueType tensorType;
    tensorType.mutable_tensortype()->set_scalartype(scalarType);
    tensorType.mutable_tensortype()->set_rank(static_cast<int64_t>(dims.size()));
    for (const auto& dim : dims) {
        tensorType.mutable_tensortype()->mutable_dimension()->Add()->CopyFrom(dim);
    }

    return tensorType;
}

V5::Value MakeBoolValue(bool b)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::BOOL));
    value.mutable_immediatevalue()->set_b(b);
    return value;
}

V5::Value MakeFloatValue(float f)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::FLOAT32));
    value.mutable_immediatevalue()->set_f(f);
    return value;
}

V5::Value MakeIntValue(int i)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::INT32));
    value.mutable_immediatevalue()->set_i(i);
    return value;
}

V5::Value MakeStringValue(std::string s)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::STRING));
    value.mutable_immediatevalue()->set_s(s);
    return value;
}

V5::Dimension MakeDim(int64_t size)
{
    V5::Dimension dim;
    dim.set_size(size);
    return dim;
}

V5::Value MakeFileValue(const std::string& file_name, uint64_t offset, const V5::ValueType& valueType)
{
    V5::Value_FileValue* fv = new V5::Value_FileValue();
    fv->set_filename(file_name);
    fv->set_offset(offset);

    V5::Value v;
    v.set_allocated_filevalue(fv);
    v.set_allocated_type(new V5::ValueType(valueType));

    return v;
}

V5::Value MakeFloatTensorValue(const std::vector<V5::Dimension>& dims, const std::vector<float>& fs)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::FLOAT32, dims));
    for (const auto& f: fs) {
        value.mutable_immediatevalue()->mutable_tensor()->mutable_floats()->Add(f);
    }
    return value;
}

V5::Value MakeIntTensorValue(const std::vector<V5::Dimension>& dims, const std::vector<int>& is)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::INT32, dims));
    for (const auto& i: is) {
        value.mutable_immediatevalue()->mutable_tensor()->mutable_ints()->Add(i);
    }
    return value;
}

V5::Block MakeBlock(const ArgBindings& bindings, const NameVec& outputs, const OpVec& ops)
{
    V5::Block block;
    for (const auto& paramAndArg : bindings) {
        (*block.mutable_inputs())[paramAndArg.first] = paramAndArg.second;
    }

    for (const auto& output : outputs){
        block.add_outputs()->append(output);
    }

    for (const auto& op : ops) {
        block.add_operations()->CopyFrom(op);
    }

    return block;
}

V5::Function MakeFunction(const NameAndTypeVec& params, const TypeVec& outputs, const V5::Block& block)
{
    V5::Function function;

    for (const auto& nameAndType : params) {
        auto input = function.add_inputs();
        input->set_name(nameAndType.first);
        input->mutable_type()->CopyFrom(nameAndType.second);
    }

    for (const auto& output : outputs) {
        function.add_outputs()->CopyFrom(output);
    }

    function.mutable_block()->CopyFrom(block);

    return function;
}

V5::Operation MakeOp(const std::string& opName, const std::string& opType, const ArgBindings& bindings,
                     const NameAndTypeVec& outputs, const Attributes& attributes)
{
    return MakeOp(opName, opType, bindings, outputs, attributes, {});
}

V5::Operation MakeOp(const std::string& opName, const std::string& opType, const ArgBindings& bindings,
                     const NameAndTypeVec& outputs, const Attributes& attributes, const BlockVec& blocks)
{
    V5::Operation op;
    op.set_name(opName);
    op.set_type(opType);

    for (const auto& paramAndArg : bindings) {
        (*op.mutable_inputs())[paramAndArg.first] = paramAndArg.second;
    }

    for (const auto& nameAndType : outputs) {
        V5::NamedValueType output;
        output.set_name(nameAndType.first);
        output.mutable_type()->CopyFrom(nameAndType.second);
        (*op.mutable_outputs()).Add()->CopyFrom(output);
    }

    for (const auto& nameAndValue : attributes) {
        (*op.mutable_attributes())[nameAndValue.first] = nameAndValue.second;
    }

    for (const auto& block : blocks) {
        op.add_blocks()->CopyFrom(block);
    }

    return op;
}

Model EmptyProgram()
{
    return ProgramWithMain(MakeFunction({}, {}, MakeBlock({}, {}, {})));
};

static ArrayFeatureType MakeArrayFeatureType(const V5::TensorType& tensorType)
{
    ArrayFeatureType aft;

    switch (tensorType.scalartype()) {
        case CoreML::Specification::V5::FLOAT32:
            aft.set_datatype(ArrayFeatureType_ArrayDataType_FLOAT32);
            break;
        case CoreML::Specification::V5::FLOAT64:
            aft.set_datatype(ArrayFeatureType_ArrayDataType_DOUBLE);
            break;
        case CoreML::Specification::V5::INT32:
            aft.set_datatype(ArrayFeatureType_ArrayDataType_INT32);
            break;
        default:
            throw std::runtime_error("Cannot create array datatype from unsupported scalar type.");
    }

    for (const auto& dim : tensorType.dimension()) {
        switch (dim.dimension_case()) {
            case CoreML::Specification::V5::Dimension::kSize:
                aft.add_shape(dim.size());
                break;
            case CoreML::Specification::V5::Dimension::kSymbol:
            case CoreML::Specification::V5::Dimension::DIMENSION_NOT_SET:
                throw std::runtime_error("Cannot create array dimension from unsupported dimension type.");
        }
    }

    return aft;
}

FeatureDescription MakeFeatureDescription(const std::string& name, const V5::ValueType& type)
{
    FeatureDescription fd;

    fd.set_name(name);

    switch (type.type_case()) {
        case V5::ValueType::kTensorType:
            fd.mutable_type()->mutable_multiarraytype()->CopyFrom(MakeArrayFeatureType(type.tensortype()));
            break;
        case V5::ValueType::kListType:
        case V5::ValueType::kScalarType:
        case V5::ValueType::kTupleType:
        case V5::ValueType::TYPE_NOT_SET:
            throw std::runtime_error("Cannot create feature from unsupported type.");
    }

    return fd;
}

FeatureDescription MakeImageFeatureDescription(const std::string& name, ImageFeatureType_ColorSpace colorSpace,
                                               int64_t height, int64_t width)
{
    FeatureDescription fd;

    fd.set_name(name);
    auto image = fd.mutable_type()->mutable_imagetype();
    image->set_colorspace(colorSpace);
    image->set_height(height);
    image->set_width(width);

    return fd;
}

static FeatureDescription MakeFeatureDescription(const V5::NamedValueType& namedType)
{
    return MakeFeatureDescription(namedType.name(), namedType.type());
}

Model ProgramWithMain(const V5::Function& main)
{
    Model model;
    model.set_specificationversion(5);
    (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);

    for (const auto& input : main.inputs()) {
        auto modelInput = MakeFeatureDescription(input);
        model.mutable_description()->add_input()->CopyFrom(modelInput);
    }

    if (main.outputs_size() != main.block().outputs_size()) {
        throw std::runtime_error("main() returns different number of values than its block.");
    }

    for (int i = 0; i < main.outputs_size(); ++i) {
        const auto& outputType = main.outputs(i);
        const auto& outputName = main.block().outputs(i);
        auto modelOutput = MakeFeatureDescription(outputName, outputType);
        model.mutable_description()->add_output()->CopyFrom(modelOutput);
    }

    return model;
}

}
