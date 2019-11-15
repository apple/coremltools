//
//  IRValueTests.cpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/15/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRValue.hpp"
#include "ILIL/IRValueType.hpp"

#include "MLModelTests.hpp"

#include "framework/TestUtils.hpp"

using namespace CoreML::ILIL;

int testIRImmediateScalarValue()
{
    auto intVal = IRScalarValue<int64_t>::Make(55);
    auto floatVal = IRScalarValue<float>::Make(5.5f);

    ML_ASSERT_EQ(IRScalarValueTypeEnum::Int64, intVal->GetType().As<IRScalarValueType>()->GetType());
    ML_ASSERT_EQ(IRScalarValueTypeEnum::Float32, floatVal->GetType().As<IRScalarValueType>()->GetType());

    ML_ASSERT_EQ(55, intVal->GetValue());
    ML_ASSERT_EQ(55, intVal->AsInt64());
    ML_ASSERT_THROWS(intVal->AsBool(), std::bad_cast);

    ML_ASSERT_EQ(5.5f, floatVal->GetValue());
    ML_ASSERT_EQ(5.5f, floatVal->AsFloat32());
    ML_ASSERT_THROWS(floatVal->AsInt64(), std::bad_cast);

    int64_t scalarInt{0};
    intVal->CopyTo(&scalarInt, sizeof(scalarInt));
    ML_ASSERT_EQ(55, scalarInt);

    float scalarFloat{0.0f};
    floatVal->CopyTo(&scalarFloat, sizeof(scalarFloat));

    return 0;
}

int testIRImmediateTensorValue()
{
    auto int64t = IRTensorValueType::Make(IRScalarValueType::Int64(), {
        IRConstantDimension::Make(1),
        IRConstantDimension::Make(4) })->Make<int64_t>({1, 2, 3, 4});

    ML_ASSERT(int64t->GetType().Is<IRTensorValueType>());
    ML_ASSERT_THROWS(int64t->AsInt64(), std::bad_cast);

    {
        std::vector<int64_t> xs;
        xs.resize(2);
        xs.shrink_to_fit();
        ML_ASSERT_THROWS(int64t->CopyTo(xs.data(), xs.size() * sizeof(int64_t)), std::runtime_error);

        xs.resize(4);
        int64t->CopyTo(xs.data(), xs.size() * sizeof(int64_t));
        ML_ASSERT_EQ(std::vector<int64_t>({1, 2, 3, 4}), xs);
    }

    return 0;
}

int testIRImmediateTupleValue()
{
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

    return 0;
}
