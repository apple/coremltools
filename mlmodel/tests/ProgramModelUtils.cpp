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

V5::Dimension MakeDim(int64_t size)
{
    V5::Dimension dim;
    dim.set_size(size);
    return dim;
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

    return op;
}

}
