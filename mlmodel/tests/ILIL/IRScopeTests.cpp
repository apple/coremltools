//
//  IRScopeTests.cpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 11/18/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRScope.hpp"
#include "ILIL/IRValue.hpp"
#include "ILIL/IRValueType.hpp"

#include "MLModelTests.hpp"

#include "framework/TestUtils.hpp"

using namespace CoreML::ILIL;

int testIRScopeGetSetType()
{
    IRScope scope(/*parentScope=*/ nullptr);
    ML_ASSERT_NULL(scope.GetParent());

    scope.SetType("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32), /*allowReplace=*/ false);
    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32), scope.GetType("anInt"));
    ML_ASSERT_NULL(scope.TryGetType("anInt", /*includeRoot=*/ false));

    scope.SetType("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt32), /*allowReplace=*/ true);
    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt32), scope.GetType("anInt"));

    ML_ASSERT_THROWS(scope.SetType("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt32), /*allowReplace=*/ false), std::runtime_error);
    ML_ASSERT_THROWS(scope.SetType("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::UInt32) /*allowReplace=false*/), std::runtime_error);

    ML_ASSERT_NULL(scope.TryGetType("aFloat"));
    ML_ASSERT_THROWS(scope.GetType("aFloat"), std::runtime_error);

    return 0;
}

int testIRScopeGetSetValue()
{
    IRScope scope(/*parentScope=*/ nullptr);
    ML_ASSERT_NULL(scope.GetParent());

    scope.SetValue("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int64)->MakeValue(int64_t{1030}), /*allowReplace=*/ false);
    ML_ASSERT_EQ(1030, scope.GetValue("anInt").AsInt64());
    ML_ASSERT_NULL(scope.TryGetValue("anInt", /*includeRoot=*/ false));

    scope.SetValue("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int64)->MakeValue(int64_t{1031}), /*allowReplace=*/ true);
    ML_ASSERT_EQ(1031, scope.GetValue("anInt").AsInt64());

    ML_ASSERT_THROWS(scope.SetValue("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int64)->MakeValue(int64_t{1032}), /*allowReplace=*/ false), std::runtime_error);
    ML_ASSERT_THROWS(scope.SetValue("anInt", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int64)->MakeValue(int64_t{1032}) /*allowReplace=false*/), std::runtime_error);

    ML_ASSERT_NULL(scope.TryGetValue("aFloat"));
    ML_ASSERT_THROWS(scope.GetValue("aFloat"), std::runtime_error);

    return 0;
}

int testIRScopeNestedTypeSearch()
{
    auto parentScope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);
    auto scope = std::make_shared<IRScope>(parentScope);
    ML_ASSERT_EQ(parentScope.get(), scope->GetParent());

    parentScope->SetType("inParent", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int8));
    parentScope->SetType("inBoth", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::BFloat16));
    scope->SetType("inBoth", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::String));
    scope->SetType("inChild", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool));

    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int8), scope->GetType("inParent"));
    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool), scope->GetType("inChild"));
    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::String), scope->GetType("inBoth"));

    ML_ASSERT_NULL(scope->TryGetType("undefined"));
    ML_ASSERT_THROWS(scope->GetType("undefined"), std::runtime_error);

    ML_ASSERT_THROWS(scope->GetType("inParent", /*searchRoot=*/ false), std::runtime_error);
    ML_ASSERT_NULL(scope->TryGetType("inParent", /*searchRoot=*/ false));

    return 0;
}

int testIRScopeNestedValueSearch()
{
    auto parentScope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);
    auto scope = std::make_shared<IRScope>(parentScope);
    ML_ASSERT_EQ(parentScope.get(), scope->GetParent());

    parentScope->SetValue("inParent", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool)->MakeValue(false));
    parentScope->SetValue("inBoth", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::String)->MakeValue<std::string>("parent"));
    scope->SetValue("inBoth", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::String)->MakeValue<std::string>("child"));
    scope->SetValue("inChild", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Float32)->MakeValue(34.3f));

    ML_ASSERT_EQ(false, scope->GetValue("inParent").AsBool());
    ML_ASSERT_EQ("child", scope->GetValue("inBoth").AsString());
    ML_ASSERT_EQ(34.3f, scope->GetValue("inChild").AsFloat32());

    ML_ASSERT_NULL(scope->TryGetType("undefined"));
    ML_ASSERT_THROWS(scope->GetType("undefined"), std::runtime_error);

    ML_ASSERT_THROWS(scope->GetValue("inParent", /*searchRoot=*/ false), std::runtime_error);
    ML_ASSERT_NULL(scope->TryGetValue("inParent", /*searchRoot=*/ false));

    return 0;
}

int testIRScopeWithRenames()
{
    auto scope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);

    scope->SetType("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool));
    scope->SetValue("x", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool)->MakeValue(false));

    scope->SetType("y", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32));
    scope->SetValue("y", IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Int32)->MakeValue(55));

    scope = scope->WithRenames({{ "x", "newX" }});

    ML_ASSERT_NULL(scope->TryGetType("x"));
    ML_ASSERT_NULL(scope->TryGetValue("x"));
    ML_ASSERT_NOT_NULL(scope->TryGetType("newX"));
    ML_ASSERT_NOT_NULL(scope->TryGetValue("newX"));
    ML_ASSERT_NOT_NULL(scope->TryGetType("y"));
    ML_ASSERT_NOT_NULL(scope->TryGetValue("y"));

    ML_ASSERT_EQ(*IRTensorValueType::MakeScalar(IRScalarValueTypeEnum::Bool), *scope->TryGetType("newX"));
    ML_ASSERT_EQ(false, scope->TryGetValue("newX")->AsBool());

    return 0;
}
