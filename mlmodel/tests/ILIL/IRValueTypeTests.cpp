//
//  IRValueTypeTests.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/15/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValueType.hpp"

#include "MLModelTests.hpp"

#include "framework/TestUtils.hpp"

using namespace CoreML::ILIL;

int testIRDimension()
{
    IRConstantDimension c1{1};
    IRConstantDimension c2{2};
    auto c1Copy = c1;

    IRSymbolicDimension s1{"1"};
    IRSymbolicDimension s2{"2"};
    auto s1Copy= s1;

    ML_ASSERT_EQ(1, c1.GetSize());
    ML_ASSERT_EQ(2, c2.GetSize());
    ML_ASSERT_EQ("1", s1.GetName());
    ML_ASSERT_EQ("2", s2.GetName());

    ML_ASSERT_EQ(c1, c1Copy);
    ML_ASSERT_NE(c1, c2);
    ML_ASSERT_EQ(s1, s1Copy);
    ML_ASSERT_NE(s1, s2);
    ML_ASSERT_NE(c1, s1);


    ML_ASSERT_NOT_NULL(c1.TryAs<IRConstantDimension>())
    ML_ASSERT_NULL(c1.TryAs<IRSymbolicDimension>());
    ML_ASSERT_NULL(s1.TryAs<IRConstantDimension>())
    ML_ASSERT_NOT_NULL(s1.TryAs<IRSymbolicDimension>());
    ML_ASSERT_THROWS(c1.As<IRSymbolicDimension>(), std::bad_cast);
    ML_ASSERT_THROWS(s1.As<IRConstantDimension>(), std::bad_cast);

    ML_ASSERT(c1.Is<IRConstantDimension>());
    ML_ASSERT(s1.Is<IRSymbolicDimension>());

    return 0;
}

int testIRScalarValueType()
{
    auto fp16 = IRScalarValueType::Float16();
    auto bfp16 = IRScalarValueType::BFloat16();

    ML_ASSERT_EQ(IRScalarValueTypeEnum::Float16, fp16->GetType());
    ML_ASSERT_EQ(1, fp16->GetNumElements());
    ML_ASSERT_EQ(IRScalarValueTypeEnum::BFloat16, bfp16->GetType());
    ML_ASSERT_EQ(1, bfp16->GetNumElements());

    ML_ASSERT_EQ(*fp16, *fp16);
    ML_ASSERT_NE(*fp16, *bfp16);

    ML_ASSERT_NULL(fp16->TryAs<IRTensorValueType>());
    ML_ASSERT_NOT_NULL(fp16->TryAs<IRScalarValueType>());

    ML_ASSERT(fp16->Is<IRScalarValueType>());
    ML_ASSERT_NOT(fp16->Is<IRTensorValueType>());

    return 0;
}

int testIRTensorValueType()
{
    auto int4t = IRTensorValueType::Make(IRScalarValueTypeEnum::Int4, {
        IRSymbolicDimension::Make("s1"),
        IRSymbolicDimension::Make("s2") });
    auto int4tAgain = IRTensorValueType::Make(IRScalarValueTypeEnum::Int4, {
        IRSymbolicDimension::Make("s1"),
        IRSymbolicDimension::Make("s2") });
    auto int8t = IRTensorValueType::Make(IRScalarValueTypeEnum::Int8, {
        IRConstantDimension::Make(2),
        IRConstantDimension::Make(4) });

    ML_ASSERT_EQ(*int4t, *int4t);
    ML_ASSERT_EQ(*int4t, *int4tAgain);
    ML_ASSERT_EQ(*int8t, *int8t);
    ML_ASSERT_NE(*int4t, *int8t);

    ML_ASSERT_EQ(IRScalarValueTypeEnum::Int4, int4t->GetScalarType());
    ML_ASSERT_EQ(IRScalarValueTypeEnum::Int8, int8t->GetScalarType());

    ML_ASSERT_EQ(8, int8t->GetNumElements());
    ML_ASSERT_THROWS(int4t->GetNumElements(), std::range_error);

    ML_ASSERT_NOT_NULL(int4t->TryAs<IRTensorValueType>());
    ML_ASSERT_NOT_NULL(int4t->As<IRTensorValueType>());
    ML_ASSERT_NULL(int8t->TryAs<IRScalarValueType>());
    ML_ASSERT_THROWS(int8t->As<IRScalarValueType>(), std::bad_cast);

    ML_ASSERT(int4t->Is<IRTensorValueType>());
    ML_ASSERT_NOT(int4t->Is<IRScalarValueType>());

    {
        auto floatType = IRTensorValueType::Make(IRScalarValueTypeEnum::Float32, { IRConstantDimension::Make(1) });
        std::vector<double> wrongTypeValues{1.0};
        std::vector<float> wrongNumValues{1.0f, 2.0f};

        ML_ASSERT_THROWS(floatType->MakeValue(std::move(wrongTypeValues)), std::runtime_error);
        ML_ASSERT_THROWS(floatType->MakeValue(std::move(wrongNumValues)), std::runtime_error);
    }

    return 0;
}

int testIRListValueType()
{
    auto lst1 = IRListValueType::Make(IRScalarValueType::Int4(), IRConstantDimension::Make(55));
    auto lst2 = IRListValueType::Make(IRScalarValueType::Int4(), IRSymbolicDimension::Make("55"));
    auto lst1Again = IRListValueType::Make(IRScalarValueType::Int4(), IRConstantDimension::Make(55));

    ML_ASSERT_EQ(*lst1, *lst1);
    ML_ASSERT_EQ(*lst1, *lst1Again);
    ML_ASSERT_NE(*lst1, *lst2);

    ML_ASSERT(lst1->GetElementType().Is<IRScalarValueType>());

    ML_ASSERT_EQ(55, lst1->GetNumElements());
    ML_ASSERT_THROWS(lst2->GetNumElements(), std::range_error);

    ML_ASSERT_NOT_NULL(lst1->TryAs<IRListValueType>());
    ML_ASSERT_NOT_NULL(lst1->As<IRListValueType>());
    ML_ASSERT_NULL(lst1->TryAs<IRScalarValueType>());
    ML_ASSERT_THROWS(lst1->As<IRScalarValueType>(), std::bad_cast);

    ML_ASSERT(lst1->Is<IRListValueType>());
    ML_ASSERT_NOT(lst1->Is<IRTupleValueType>());

    return 0;
}

int testIRTupleValueType()
{
    auto tup1 = IRTupleValueType::Make(IRTupleValueType::ValueTypePtrVec{
        IRListValueType::Make(IRScalarValueType::UInt16(), IRConstantDimension::Make(400)),
        IRScalarValueType::Bool()
    });

    auto tup1Again = IRTupleValueType::Make(IRTupleValueType::ValueTypePtrVec{
        IRListValueType::Make(IRScalarValueType::UInt16(), IRConstantDimension::Make(400)),
        IRScalarValueType::Bool()
    });

    auto tup2 = IRTupleValueType::Make(IRTupleValueType::ValueTypePtrVec{
        IRListValueType::Make(IRScalarValueType::UInt64(), IRConstantDimension::Make(400)),
        IRScalarValueType::Bool()
    });

    ML_ASSERT_EQ(*tup1, *tup1);
    ML_ASSERT_EQ(*tup1, *tup1Again);
    ML_ASSERT_NE(*tup1, *tup2);

    ML_ASSERT(tup1->GetTypes()[0]->Is<IRListValueType>());
    ML_ASSERT(tup1->GetTypes()[1]->Is<IRScalarValueType>());

    ML_ASSERT_THROWS(tup1->GetNumElements(), std::range_error);

    ML_ASSERT(tup1->Is<IRTupleValueType>());
    ML_ASSERT_NOT_NULL(tup1->TryAs<IRTupleValueType>());
    ML_ASSERT_NOT_NULL(tup1->As<IRTupleValueType>());
    ML_ASSERT_NULL(tup1->TryAs<IRListValueType>());
    ML_ASSERT_THROWS(tup1->As<IRListValueType>(), std::bad_cast);

    auto boolType = IRScalarValueType::Bool();
    auto int64Type = IRScalarValueType::Int64();

    auto boolAndint64Type = IRTupleValueType::Make({ boolType, int64Type });
    auto boolAndint64 = boolAndint64Type->Make({
        boolType->Make(false),
        int64Type->Make(int64_t{97})
    });

    ML_ASSERT(boolAndint64->GetType().Is<IRTupleValueType>());
    ML_ASSERT_THROWS(boolAndint64->AsBool(), std::bad_cast);

    ML_ASSERT_EQ(false, boolAndint64->GetValues().at(0)->AsBool());
    ML_ASSERT_EQ(97, boolAndint64->GetValues().at(1)->AsInt64());

    {
        IRTupleValueType::ConstIRValueVec wrongTypeValues{
            boolType->Make(false),
            IRScalarValueType::Int32()->Make(97)
        };
        IRTupleValueType::ConstIRValueVec wrongNumValues{ };
        ML_ASSERT_THROWS(boolAndint64Type->Make(std::move(wrongTypeValues)), std::runtime_error);
        ML_ASSERT_THROWS(boolAndint64Type->Make(std::move(wrongNumValues)), std::runtime_error);
    }


    return 0;
}

int testIRNamedValueType()
{
    auto nvt = IRNamedValueType::Make("aDynamic", IRScalarValueType::Dynamic());

    ML_ASSERT_EQ("aDynamic", nvt->GetName());
    ML_ASSERT(nvt->GetType().Is<IRScalarValueType>());

    return 0;
}
