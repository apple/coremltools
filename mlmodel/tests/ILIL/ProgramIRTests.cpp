//
//  ProgramIRTests.cpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 11/18/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "MLModelTests.hpp"

#include "ProgramModelUtils.hpp"
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
using namespace ::ProgramModelUtils;

int testParseProgramIRBlock()
{
    // [Parent scope] string a <- "relu";
    // [Parent scope] fp[2] b <- ... ;
    // [x <- a] { z = relu(x=x); } [z]
    auto block = MakeBlock({ { "x", "a" } }, // inputs
                           { "z" }, // outputs
                           { // ops
        MakeOp("z", "relu",
               { { "x", "x" } }, // inputs
               { { "z", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
               {})
    });

    auto parentScope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);
    auto tensorType = IRTensorValueType::Make(IRScalarValueTypeEnum::Float32, { std::make_shared<IRConstantDimension>(2)});
    parentScope->SetType("a", tensorType);

    auto irBlock = ProgramIRBlock::Parse(block, parentScope);

    ML_ASSERT_EQ("a", irBlock->GetArgumentName("x"))
    ML_ASSERT_EQ(1, irBlock->GetInputs().size());
    ML_ASSERT_EQ("a", irBlock->GetInputs().at("x"));
    ML_ASSERT_EQ(1, irBlock->GetOperations().size());
    ML_ASSERT_EQ(1, irBlock->GetOutputs().size());
    ML_ASSERT_EQ("z", irBlock->GetOutputs()[0]);

    ML_ASSERT_EQ(*tensorType, irBlock->GetScope().GetType("x"));

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

    std::string fuckyou = "x";
    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int4), irFunc->GetInputType(fuckyou));

    ML_ASSERT_EQ(1, irFunc->GetOutputTypes().size());
    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int4), *irFunc->GetOutputTypes()[0]);

    return 0;
}

int testParseProgramIROperation()
{
    auto op = MakeOp("anOp", "const",
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

    ML_ASSERT_EQ(1, irProgram->GetParameterNames().size());
    ML_ASSERT_EQ("aBool", irProgram->GetParameterNames()[0]);
    auto irTensorTypePtr = irProgram->GetScope().GetType("aBool").TryAs<const IRTensorValueType>();
    ML_ASSERT_NOT_NULL(irTensorTypePtr);
    ML_ASSERT_EQ(IRScalarValueTypeEnum::Bool, irTensorTypePtr->GetScalarType());
    ML_ASSERT_EQ(true, irProgram->GetScope().GetValue("aBool").AsBool());
    ML_ASSERT_EQ(true, irProgram->GetParameterValue("aBool").AsBool());

    ML_ASSERT_EQ(1, irProgram->GetFunctions().size());
    irTensorTypePtr = irProgram->GetFunction("main").GetOutputTypes()[0]->TryAs<const IRTensorValueType>();
    ML_ASSERT_NOT_NULL(irTensorTypePtr);
    ML_ASSERT_EQ(IRScalarValueTypeEnum::Bool, irTensorTypePtr->GetScalarType());

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
    
    // Immediate bool value
    {
        auto irValue = ProgramIRValue::Parse(MakeBoolValue(false));
        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool), irValue->GetType());
        ML_ASSERT_EQ(false, irValue->AsBool());
    }
    
    // Immediate float value
    {
        auto irValue = ProgramIRValue::Parse(MakeFloatValue(8.24f));
        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Float32), irValue->GetType());
        ML_ASSERT_EQ(8.24f, irValue->AsFloat32());
    }

    // Immediate int32 value
    {
        auto irValue = ProgramIRValue::Parse(MakeIntValue(74));
        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32), irValue->GetType());
        ML_ASSERT_EQ(74, irValue->AsInt32());
    }

    // Immediate string value
    {
        auto irValue = ProgramIRValue::Parse(MakeStringValue("abc123"));
        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::String), irValue->GetType());
        ML_ASSERT_EQ("abc123", irValue->AsString());
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
            ML_ASSERT_EQ(protoAndIr.second, irValueType->As<IRTensorValueType>()->GetScalarType());
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

        ML_ASSERT_EQ(IRScalarValueTypeEnum::BFloat16, irTensorType->GetScalarType());
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

        ML_ASSERT_EQ(IRScalarValueTypeEnum::Dynamic, irTensorType->GetScalarType());
        ML_ASSERT_EQ(2, irTensorType->GetShape().size());
        ML_ASSERT_EQ(4, irTensorType->GetShape()[0]->As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ(2, irTensorType->GetShape()[1]->As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ(8, irTensorType->GetNumElements());
    }

    // List type with constant length
    {
        V5::ValueType listType;
        listType.mutable_listtype()->mutable_type()->mutable_tensortype()->set_scalartype(V5::ScalarType::UINT64);
        listType.mutable_listtype()->mutable_length()->set_size(102);
        auto irValueType = ProgramIRValueType::Parse(listType);
        auto irListType = irValueType->As<IRListValueType>();

        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt64), irListType->GetElementType());
        ML_ASSERT_EQ(102, irListType->GetLength().As<IRConstantDimension>()->GetSize());
        ML_ASSERT_EQ(102, irListType->GetNumElements());
    }

    // List type with symbolic length
    {
        V5::ValueType listType;
        listType.mutable_listtype()->mutable_type()->mutable_tensortype()->set_scalartype(V5::ScalarType::UINT64);
        listType.mutable_listtype()->mutable_length()->set_symbol("s3");
        auto irValueType = ProgramIRValueType::Parse(listType);
        auto irListType = irValueType->As<IRListValueType>();

        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt64), irListType->GetElementType());
        ML_ASSERT_EQ("s3", irListType->GetLength().As<IRSymbolicDimension>()->GetName());
        ML_ASSERT_THROWS(irListType->GetNumElements(), std::range_error); // can't compute length from with symbolic dimension
    }

    return 0;
}

int testProgramIRBlockRename()
{
    // Input Params
    {
        auto scope = std::make_shared<IRScope>(nullptr);
        scope->SetType("a", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int8));
        IROperation::IRBlockPtr block = ProgramIRBlock::Make(scope, { { "x", "a" } }, {}, {});
        block = block->WithRenames({ { "x", "notX" } });

        ML_ASSERT_EQ(size_t(0), block->GetInputs().count("x"));
        ML_ASSERT_EQ(size_t(1), block->GetInputs().count("notX"));
    }

    // Input Args
    {
        auto scope = std::make_shared<IRScope>(nullptr);
        scope->SetType("a", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int8));
        IROperation::IRBlockPtr block = ProgramIRBlock::Make(scope, { { "x", "a" } }, {}, {});
        block = block->WithRenames({ {"a", "notA" } });

        ML_ASSERT_EQ("notA", block->GetInputs().at("x"));
        ML_ASSERT_NULL(block->GetScope().TryGetType("a"));
        ML_ASSERT_NOT_NULL(block->GetScope().TryGetType("notA"));
    }

    // Outputs
    {
        auto scope = std::make_shared<IRScope>(nullptr);
        scope->SetType("a", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int8));
        IROperation::IRBlockPtr block = ProgramIRBlock::Make(scope, {}, { "a" }, {});
        block = block->WithRenames({{ "a", "notA" }});

        ML_ASSERT_EQ("notA", block->GetOutputs().at(0));
    }

    // Ops
    {
        auto scope = std::make_shared<IRScope>(nullptr);
        scope->SetType("a", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int8));
        auto op = ProgramIROperation::Make(scope, "anOp", "const", {}, {}, { "a" }, {});
        IROperation::IRBlockPtr block = ProgramIRBlock::Make(scope, {}, {}, { std::move(op) });
        block = block->WithRenames({{ "a", "notA" }});

        ML_ASSERT_EQ("notA", block->GetOperations().at(0)->GetOutputNames().at(0));
    }

    return 0;
}

int testProgramIRFunctionRename()
{
    // inputs and scope
    {
        auto funcScope = std::make_shared<IRScope>(nullptr);
        funcScope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt8));

        auto blockScope = std::make_shared<IRScope>(funcScope);
        auto block = ProgramIRBlock::Make(blockScope, {}, {}, {});

        auto origFunc = ProgramIRFunction::Make({{ "x", funcScope->GetTypeSharedPtr("x") }},
                                                {}, funcScope, std::move(block));
        auto renamedFunc = origFunc->WithRenames({{ "x", "notX" }});

        ML_ASSERT_EQ(size_t(1), renamedFunc->GetInputs().size());
        ML_ASSERT_EQ(size_t(1), renamedFunc->GetInputs().count("notX"));

        ML_ASSERT_NULL(renamedFunc->GetScope().TryGetType("x"));
        ML_ASSERT_NOT_NULL(renamedFunc->GetScope().TryGetType("notX"));
    }

    // block
    {
        auto funcScope = std::make_shared<IRScope>(nullptr);
        auto blockScope = std::make_shared<IRScope>(funcScope);
        blockScope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt8));

        auto block = ProgramIRBlock::Make(blockScope, {}, {}, {});

        auto origFunc = ProgramIRFunction::Make({}, {}, funcScope, std::move(block));
        auto renamedFunc = origFunc->WithRenames({{ "x", "notX" }});

        ML_ASSERT_NULL(renamedFunc->GetBlock().GetScope().TryGetType("x"));
        ML_ASSERT_NOT_NULL(renamedFunc->GetBlock().GetScope().TryGetType("notX"));
    }

    return 0;
}

int testProgramIROperationRename()
{
    auto makeScope = []() {
        auto scope = std::make_shared<IRScope>(nullptr);
        scope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::BFloat16));
        return scope;
    };

    { // Rename input args
        auto origScope = makeScope();
        auto origOp = ProgramIROperation::Make(origScope, "anOp", "const", {},
                                               { { "param1", "x" } }, {}, {});

        ProgramIROperation::RenameVec renames{{ "x", "notX", }};
        ProgramIROperation::ScopePtr renamedScope = origScope->WithRenames(renames);
        auto renamedOp = origOp->WithRenames(renames, renamedScope);

        ML_ASSERT_EQ("notX", renamedOp->GetInput("param1"));
        ML_ASSERT_EQ("notX", renamedOp->GetInputNames().at(0));
        ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::BFloat16), renamedOp->GetScope().GetType("notX"));
        ML_ASSERT_NULL(renamedOp->GetScope().TryGetType("x"));
    }

    { // Rename output
        auto origScope = makeScope();
        auto origOp = ProgramIROperation::Make(origScope, "anOp", "const", {},
                                               {}, { "x" }, {});

        ProgramIROperation::RenameVec renames{{ "x", "notX", }};
        ProgramIROperation::ScopePtr renamedScope = origScope->WithRenames(renames);
        auto renamedOp = origOp->WithRenames(renames, renamedScope);

        ML_ASSERT_EQ("notX", renamedOp->GetOutputNames().at(0));
    }

    { // Rename in child block
        // {
        //   x = ...
        //   _ = outerOp{    # Call rename (x -> notX) on this
        //     x = innerOp()
        //   }
        // }
        auto outerScope = std::make_shared<IRScope>(nullptr);
        auto innerScope = std::make_shared<IRScope>(outerScope);
        outerScope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::BFloat16));

        auto innerOp = ProgramIROperation::Make(innerScope, "innerOp", "const", {},
                                              {}, { "x" }, {});
        auto innerBlock = ProgramIRBlock::Make(innerScope, {}, {}, { std::move(innerOp) });

        auto op = ProgramIROperation::Make(outerScope, "outerOp", "const",
                                           {}, {}, {}, { std::move(innerBlock) });

        ProgramIROperation::RenameVec renames{{ "x", "notX", }};
        ProgramIROperation::ScopePtr renamedScope = outerScope->WithRenames(renames);
        auto renamedOp = op->WithRenames({{ "x", "notX" }}, renamedScope);

        ML_ASSERT_EQ("notX", renamedOp->GetBlocks().at(0)->GetOperations().at(0)->GetOutputNames().at(0));
    }

    return 0;
}

int testProgramIRProgramRename()
{
    // parameters, parameter names, scope
    {
        auto scope = std::make_shared<IRScope>(nullptr);
        scope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32));

        auto origProgram = ProgramIRProgram::Make({{ "x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32)->MakeValue(56) }}, scope, {});
        auto renamedProgram = origProgram->WithRenames({{ "x", "notX" }});

        ML_ASSERT_EQ(size_t(1), renamedProgram->GetParameters().size());
        ML_ASSERT_EQ(size_t(1), renamedProgram->GetParameters().count("notX"));
        ML_ASSERT_EQ(size_t(1), renamedProgram->GetParameterNames().size());
        ML_ASSERT_EQ("notX", renamedProgram->GetParameterNames().at(0));
        ML_ASSERT_NULL(renamedProgram->GetScope().TryGetType("x"));
        ML_ASSERT_NOT_NULL(renamedProgram->GetScope().TryGetType("notX"));
    }

    // functions
    {
        auto programScope = std::make_shared<IRScope>(nullptr);

        auto funcScope = std::make_shared<IRScope>(programScope);
        funcScope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt8));

        auto blockScope = std::make_shared<IRScope>(funcScope);
        auto block = ProgramIRBlock::Make(blockScope, {}, {}, {});

        auto func = ProgramIRFunction::Make({{ "x", funcScope->GetTypeSharedPtr("x") }},
                                            {}, funcScope, std::move(block));

        auto origProgram = ProgramIRProgram::Make({}, programScope, {{ "main", std::move(func) }});
        auto renamedProgram = origProgram->WithRenames({{ "x", "notX" }});

        ML_ASSERT_EQ(size_t(1), renamedProgram->GetFunction("main").GetInputs().count("notX"));
    }

    return 0;
}
