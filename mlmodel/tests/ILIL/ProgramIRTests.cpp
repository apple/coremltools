//
//  ProgramIRTests.cpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 11/18/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "MLModelTests.hpp"

#include "Format.hpp"
#include "framework/TestUtils.hpp"
#include "ILIL/Program/ProgramIRBlock.hpp"
#include "ILIL/Program/ProgramIRFunction.hpp"
#include "ILIL/Program/ProgramIROperation.hpp"
#include "ILIL/Program/ProgramIRProgram.hpp"
#include "ILIL/Program/ProgramIRValue.hpp"
#include "ILIL/Program/ProgramIRValueType.hpp"

#include <array>

using namespace ::CoreML::ILIL;
using namespace ::CoreML::ILIL::Program;
using namespace ::CoreML::Specification;

using ArgBindings = std::vector<std::pair<std::string, std::string>>;
using Attributes = std::vector<std::pair<std::string, V5::Value>>;
using NameAndTypeVec = std::vector<std::pair<std::string, V5::ValueType>>;
using NameVec = std::vector<std::string>;
using OpVec = std::vector<V5::Operation>;
using TypeVec = std::vector<V5::ValueType>;

static V5::ValueType MakeScalarValueType(V5::ScalarType scalarType)
{
    V5::ValueType valueType;
    valueType.set_scalartype(scalarType);
    return valueType;
}

static V5::Value MakeBoolValue(bool b)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::BOOL));
    value.mutable_immediatevalue()->set_b(b);
    return value;
}

static V5::Value MakeFloatValue(float f)
{
    V5::Value value;
    value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::FLOAT32));
    value.mutable_immediatevalue()->set_f(f);
    return value;
}

static V5::Dimension MakeDim(int64_t size)
{
    V5::Dimension dim;
    dim.set_size(size);
    return dim;
}

static V5::ValueType MakeTensorValueType(V5::ScalarType scalarType, const std::vector<V5::Dimension>& dims)
{
    V5::ValueType tensorType;
    tensorType.mutable_tensortype()->set_scalartype(scalarType);
    tensorType.mutable_tensortype()->set_rank(static_cast<int64_t>(dims.size()));
    for (const auto& dim : dims) {
        tensorType.mutable_tensortype()->mutable_dimension()->Add()->CopyFrom(dim);
    }

    return tensorType;
}

static V5::Block MakeBlock(const ArgBindings& bindings, const NameVec& outputs, const OpVec& ops)
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

static V5::Function MakeFunction(const NameAndTypeVec& params, const TypeVec& outputs, const V5::Block& block)
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

static V5::Operation MakeOp(const std::string& opName, const std::string& opType, const ArgBindings& bindings,
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

int testParseProgramIRBlock()
{
    // [Parent scope] string a <- "relu";
    // [Parent scope] fp[2] b <- ... ;
    // [x <- a, y <- b] { z = activation(data=y, activation=x); } [z]
    auto block = MakeBlock({ { "x", "a" }, { "y", "b" } }, // inputs
                           { "z" }, // outputs
                           { // ops
        MakeOp("z", "Activation",
               { { "activation", "a" }, { "data", "b" } }, // inputs
               { { "z", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
               {})
    });

    auto parentScope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);
    parentScope->SetType("a", IRScalarValueType::String());
    parentScope->SetValue("a", IRScalarValueType::String()->Make<std::string>("relu"));
    auto tensorType = IRTensorValueType::Make(IRScalarValueType::Float32(), { std::make_shared<IRConstantDimension>(2)} );
    parentScope->SetType("b", tensorType);

    auto irBlock = ProgramIRBlock::Parse(block, parentScope);

    ML_ASSERT_EQ("a", irBlock->GetArgumentName("x"))
    ML_ASSERT_EQ("b", irBlock->GetArgumentName("y"))
    ML_ASSERT_EQ(1, irBlock->GetOperations().size());
    ML_ASSERT_EQ(1, irBlock->GetOutputs().size());
    ML_ASSERT_EQ("z", irBlock->GetOutputs()[0]);

    ML_ASSERT_EQ(*IRScalarValueType::String(), *irBlock->GetScope().GetType("x"));
    ML_ASSERT_EQ("relu", irBlock->GetScope().GetValue("x")->AsString());
    ML_ASSERT_EQ(*tensorType, *irBlock->GetScope().GetType("y"));

    return 0;
}

int testParseProgramIRFunction()
{
    // f(int4 x)
    //   return x
    auto function = MakeFunction({{ "x", MakeScalarValueType(V5::ScalarType::INT4)}},
                                 { MakeScalarValueType(V5::ScalarType::INT4) },
                                 MakeBlock({}, {"x"}, {}));

    auto parentScope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);
    auto irFunc = ProgramIRFunction::Parse(function, parentScope);

    ML_ASSERT_EQ(1, irFunc->GetBlock().GetOutputs().size());
    ML_ASSERT_EQ("x", irFunc->GetBlock().GetOutputs()[0]);

    ML_ASSERT_EQ(*IRScalarValueType::Int4(), irFunc->GetInputType("x"));

    ML_ASSERT_EQ(1, irFunc->GetOutputTypes().size());
    ML_ASSERT_EQ(*IRScalarValueType::Int4(), *irFunc->GetOutputTypes()[0]);

    return 0;
}

int testParseProgramIROperation()
{
    auto op = MakeOp("anOp", "Const",
                     { { "param1", "arg1" }, { "param2", "arg2" } },
                     { { "output1", MakeScalarValueType(V5::ScalarType::BOOL) },
                       { "output2", MakeScalarValueType(V5::ScalarType::DYNAMIC) } },
                     { { "value", MakeBoolValue(false) } });

    auto scope = std::shared_ptr<IRScope>(/*parentScope=*/ nullptr);
    auto irOp = ProgramIROperation::Parse(op, scope);

    ML_ASSERT_EQ(false, irOp->GetAttribute("value").AsBool());
    ML_ASSERT_THROWS(irOp->GetAttribute("doesNotExist"), std::out_of_range);

    ML_ASSERT(irOp->GetBlocks().empty());

    ML_ASSERT_EQ(2, irOp->GetNumInputs());
    ML_ASSERT_EQ("arg1", irOp->GetInputNames()[0]);
    ML_ASSERT_EQ("arg2", irOp->GetInputNames()[1]);
    ML_ASSERT_EQ("arg1", irOp->GetInput("param1"));
    ML_ASSERT_EQ("arg2", irOp->GetInput("param2"));

    ML_ASSERT_EQ(2, irOp->GetNumOutputs());
    ML_ASSERT_EQ("output1", irOp->GetOutputNames()[0]);
    ML_ASSERT_EQ("output2", irOp->GetOutputNames()[1]);

    return 0;
}

int testParseProgramIRProgram()
{
    V5::Program program;

    (*program.mutable_parameters())["aBool"] = MakeBoolValue(true);

    // f()
    //   return aBool
    (*program.mutable_functions())["main"] = MakeFunction({},
                                                          {MakeScalarValueType(V5::ScalarType::BOOL)},
                                                          MakeBlock({}, {"aBool"}, {}));

    auto irProgram = ProgramIRProgram::Parse(program);

    ML_ASSERT_EQ(*IRScalarValueType::Bool(), *irProgram->GetScope().GetType("aBool"));
    ML_ASSERT_EQ(true, irProgram->GetScope().GetValue("aBool")->AsBool());
    ML_ASSERT_EQ(true, irProgram->GetParameterValue("aBool").AsBool());

    ML_ASSERT_EQ(1, irProgram->GetFunctions().size());
    ML_ASSERT_EQ(*IRScalarValueType::Bool(), *irProgram->GetFunction("main").GetOutputTypes()[0]);

    return 0;
}

int testParseProgramIRScalarValue()
{
    // Immediate float value
    {
        auto irValue = ProgramIRValue::Parse(MakeFloatValue(8.24f));
        ML_ASSERT_EQ(*IRScalarValueType::Float32(), irValue->GetType());
        ML_ASSERT_EQ(8.24f, irValue->AsFloat32());
    }

    // Immediate int64 value
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::INT64));
        value.mutable_immediatevalue()->set_i(74);
        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT_EQ(*IRScalarValueType::Int64(), irValue->GetType());
        ML_ASSERT_EQ(74, irValue->AsInt64());
    }

    // Immediate bool value
    {
        auto irValue = ProgramIRValue::Parse(MakeBoolValue(false));
        ML_ASSERT_EQ(*IRScalarValueType::Bool(), irValue->GetType());
        ML_ASSERT_EQ(false, irValue->AsBool());
    }

    // Immediate bool value
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeScalarValueType(V5::ScalarType::STRING));
        value.mutable_immediatevalue()->set_s("hi there");
        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT_EQ(*IRScalarValueType::String(), irValue->GetType());
        ML_ASSERT_EQ("hi there", irValue->AsString());
    }

    return 0;
}

int testParseProgramIRTensorValue()
{
    // Immediate float tensor
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(3) }));
        std::array<float, 6> expected = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
        for (const auto x : expected) {
            value.mutable_immediatevalue()->mutable_tensor()->mutable_floats()->Add(x);
        }

        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT(irValue->GetType().Is<IRTensorValueType>());

        std::array<float, 6> actual;
        irValue->CopyTo(&actual, sizeof(actual));
        ML_ASSERT_EQ(expected, actual);
    }

    // Immediate int64 tensor
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::INT64, { MakeDim(2), MakeDim(3) }));
        std::array<int64_t, 6> expected = { 1, 2, 3, 4, 5, 6 };
        for (const auto x : expected) {
            value.mutable_immediatevalue()->mutable_tensor()->mutable_ints()->Add(x);
        }

        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT(irValue->GetType().Is<IRTensorValueType>());

        std::array<int64_t, 6> actual;
        irValue->CopyTo(&actual, sizeof(actual));
        ML_ASSERT_EQ(expected, actual);
    }

    // Immediate int64 tensor
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::INT64, { MakeDim(2), MakeDim(3) }));
        std::array<int64_t, 6> expected = { 1, 2, 3, 4, 5, 6 };
        for (const auto x : expected) {
            value.mutable_immediatevalue()->mutable_tensor()->mutable_ints()->Add(x);
        }

        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT(irValue->GetType().Is<IRTensorValueType>());

        std::array<int64_t, 6> actual;
        irValue->CopyTo(&actual, sizeof(actual));
        ML_ASSERT_EQ(expected, actual);
    }

    // Immediate bool tensor
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::BOOL, { MakeDim(2) }));
        std::array<bool, 2> expected = { true, false };
        for (const auto x : expected) {
            value.mutable_immediatevalue()->mutable_tensor()->mutable_bools()->Add(x);
        }

        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT(irValue->GetType().Is<IRTensorValueType>());

        // We haven't decided on an in-memory boolean vector representation so we throw on copy.
        std::array<bool, 2> actual;
        ML_ASSERT_THROWS(irValue->CopyTo(&actual, sizeof(actual)), std::runtime_error);
    }

    // Immediate string tensor
    {
        V5::Value value;
        value.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::STRING, { MakeDim(2) }));
        std::array<std::string, 2> expected = { "hello", "there" };
        for (const auto& x : expected) {
            value.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->Add(x.c_str());
        }

        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT(irValue->GetType().Is<IRTensorValueType>());

        // We haven't decided on an in-memory string vector representation so we throw on copy.
        std::array<std::string, 2> actual;
        ML_ASSERT_THROWS(irValue->CopyTo(&actual, sizeof(actual)), std::runtime_error);
    }

    return 0;
}

int testParseProgramIRTupleValue()
{
    // bool, float32 tuple
    {
        V5::Value value;
        value.mutable_type()->mutable_tupletype()->add_values()->CopyFrom(MakeScalarValueType(V5::BOOL));
        value.mutable_type()->mutable_tupletype()->add_values()->CopyFrom(MakeScalarValueType(V5::FLOAT32));
        value.mutable_immediatevalue()->mutable_tuple()->add_value()->CopyFrom(MakeBoolValue(true));
        value.mutable_immediatevalue()->mutable_tuple()->add_value()->CopyFrom(MakeFloatValue(9.37f));

        auto irValue = ProgramIRValue::Parse(value);
        ML_ASSERT(irValue->GetType().Is<IRTupleValueType>());

        std::tuple<bool, float> actual;
        // We haven't decided on an in-memory tuple representation so we throw on copy.
        ML_ASSERT_THROWS(irValue->CopyTo(&actual, sizeof(actual)), std::runtime_error);
    }

    return 0;
}

int testParseProgramIRValueType()
{
    // All scalar types
    {
        std::vector<std::pair<V5::ScalarType, IRScalarValueTypeEnum>> cases = {
            { V5::ScalarType::DYNAMIC, IRScalarValueTypeEnum::Dynamic},
            { V5::ScalarType::BOOL, IRScalarValueTypeEnum::Bool},
            { V5::ScalarType::STRING, IRScalarValueTypeEnum::String},
            { V5::ScalarType::FLOAT16, IRScalarValueTypeEnum::Float16},
            { V5::ScalarType::FLOAT32, IRScalarValueTypeEnum::Float32},
            { V5::ScalarType::FLOAT64, IRScalarValueTypeEnum::Float64},
            { V5::ScalarType::BFLOAT16, IRScalarValueTypeEnum::BFloat16},
            { V5::ScalarType::INT4, IRScalarValueTypeEnum::Int4},
            { V5::ScalarType::INT8, IRScalarValueTypeEnum::Int8},
            { V5::ScalarType::INT16, IRScalarValueTypeEnum::Int16},
            { V5::ScalarType::INT32, IRScalarValueTypeEnum::Int32},
            { V5::ScalarType::INT64, IRScalarValueTypeEnum::Int64},
            { V5::ScalarType::UINT4, IRScalarValueTypeEnum::UInt4},
            { V5::ScalarType::UINT8, IRScalarValueTypeEnum::UInt8},
            { V5::ScalarType::UINT16, IRScalarValueTypeEnum::UInt16},
            { V5::ScalarType::UINT32, IRScalarValueTypeEnum::UInt32},
            { V5::ScalarType::UINT64, IRScalarValueTypeEnum::UInt64}
        };

        for (const auto& protoAndIr : cases) {
            V5::ValueType specValueType = MakeScalarValueType(protoAndIr.first);
            auto irValueType = ProgramIRValueType::Parse(specValueType);
            ML_ASSERT_EQ(protoAndIr.second, irValueType->As<IRScalarValueType>()->GetType());
        }
    }

    // Tensor type with symbolic dimension
    {
        V5::ValueType tensorType;
        tensorType.mutable_tensortype()->set_scalartype(V5::ScalarType::BFLOAT16);
        tensorType.mutable_tensortype()->set_rank(2);
        tensorType.mutable_tensortype()->mutable_dimension()->Add()->set_size(4);
        tensorType.mutable_tensortype()->mutable_dimension()->Add()->set_symbol("s1");
        auto irValueType = ProgramIRValueType::Parse(tensorType);
        auto irTensorType = irValueType->As<IRTensorValueType>();

        ML_ASSERT_EQ(*IRScalarValueType::BFloat16(), irTensorType->GetScalarType());
        ML_ASSERT_EQ(2, irTensorType->GetShape().size());
        ML_ASSERT_EQ(4, irTensorType->GetShape()[0]->As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ("s1", irTensorType->GetShape()[1]->As<IRSymbolicDimension>()->GetName());
        ML_ASSERT_THROWS(irTensorType->GetNumElements(), std::range_error); // can't compute size from with symbolic dimension
    }

    // Tensor type with all constant dimensions
    {
        V5::ValueType tensorType;
        tensorType.mutable_tensortype()->set_scalartype(V5::ScalarType::DYNAMIC);
        tensorType.mutable_tensortype()->set_rank(2);
        tensorType.mutable_tensortype()->mutable_dimension()->Add()->set_size(4);
        tensorType.mutable_tensortype()->mutable_dimension()->Add()->set_size(2);
        auto irValueType = ProgramIRValueType::Parse(tensorType);
        auto irTensorType = irValueType->As<IRTensorValueType>();

        ML_ASSERT_EQ(*IRScalarValueType::Dynamic(), irTensorType->GetScalarType());
        ML_ASSERT_EQ(2, irTensorType->GetShape().size());
        ML_ASSERT_EQ(4, irTensorType->GetShape()[0]->As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ(2, irTensorType->GetShape()[1]->As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ(8, irTensorType->GetNumElements());
    }

    // List type with constant length
    {
        V5::ValueType listType;
        listType.mutable_listtype()->mutable_type()->set_scalartype(V5::ScalarType::UINT64);
        listType.mutable_listtype()->mutable_length()->set_size(102);
        auto irValueType = ProgramIRValueType::Parse(listType);
        auto irListType = irValueType->As<IRListValueType>();

        ML_ASSERT_EQ(*IRScalarValueType::UInt64(), *irListType->GetElementType().As<IRScalarValueType>());
        ML_ASSERT_EQ(102, irListType->GetLength().As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ(102, irListType->GetNumElements());
    }

    // List type with symbolic length
    {
        V5::ValueType listType;
        listType.mutable_listtype()->mutable_type()->set_scalartype(V5::ScalarType::UINT64);
        listType.mutable_listtype()->mutable_length()->set_symbol("s3");
        auto irValueType = ProgramIRValueType::Parse(listType);
        auto irListType = irValueType->As<IRListValueType>();

        ML_ASSERT_EQ(*IRScalarValueType::UInt64(), *irListType->GetElementType().As<IRScalarValueType>());
        ML_ASSERT_EQ("s3", irListType->GetLength().As<IRSymbolicDimension>()->GetName());
        ML_ASSERT_THROWS(irListType->GetNumElements(), std::range_error); // can't compute length from with symbolic dimension
    }

    return 0;
}
