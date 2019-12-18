//
//  IRScopeTests.cpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 11/18/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IRScope.hpp"

#include "ILIL/IRValueType.hpp"

#include "MLModelTests.hpp"

#include "framework/TestUtils.hpp"

using namespace CoreML::ILIL;

int testIRScopeGetSetType()
{
    IRScope scope(/*parentScope=*/ nullptr);
    ML_ASSERT_NULL(scope.GetParent());

    scope.SetType("anInt", IRScalarValueType::Int32(), /*allowReplace=*/ false);
    ML_ASSERT_EQ(*IRScalarValueType::Int32(), *scope.GetType("anInt"));
    ML_ASSERT_NULL(scope.TryGetType("anInt", /*includeRoot=*/ false));

    scope.SetType("anInt", IRScalarValueType::UInt32(), /*allowReplace=*/ true);
    ML_ASSERT_EQ(*IRScalarValueType::UInt32(), *scope.GetType("anInt"));

    ML_ASSERT_THROWS(scope.SetType("anInt", IRScalarValueType::UInt32(), /*allowReplace=*/ false), std::runtime_error);
    ML_ASSERT_THROWS(scope.SetType("anInt", IRScalarValueType::UInt32() /*allowReplace=false*/), std::runtime_error);

    ML_ASSERT_NULL(scope.TryGetType("aFloat"));
    ML_ASSERT_THROWS(scope.GetType("aFloat"), std::runtime_error);

    return 0;
}

int testIRScopeGetSetValue()
{
    IRScope scope(/*parentScope=*/ nullptr);
    ML_ASSERT_NULL(scope.GetParent());

    scope.SetValue("anInt", IRScalarValueType::Int64()->Make(int64_t{1030}), /*allowReplace=*/ false);
    ML_ASSERT_EQ(1030, scope.GetValue("anInt")->AsInt64());
    ML_ASSERT_NULL(scope.TryGetValue("anInt", /*includeRoot=*/ false));

    scope.SetValue("anInt", IRScalarValueType::Int64()->Make(int64_t{1031}), /*allowReplace=*/ true);
    ML_ASSERT_EQ(1031, scope.GetValue("anInt")->AsInt64());

    ML_ASSERT_THROWS(scope.SetValue("anInt", IRScalarValueType::Int64()->Make(int64_t{1032}), /*allowReplace=*/ false), std::runtime_error);
    ML_ASSERT_THROWS(scope.SetValue("anInt", IRScalarValueType::Int64()->Make(int64_t{1032}) /*allowReplace=false*/), std::runtime_error);

    ML_ASSERT_NULL(scope.TryGetValue("aFloat"));
    ML_ASSERT_THROWS(scope.GetValue("aFloat"), std::runtime_error);

    return 0;
}

int testIRScopeNestedTypeSearch()
{
    auto parentScope = std::make_shared<IRScope>(/*parentScope=*/ nullptr);
    auto scope = std::make_shared<IRScope>(parentScope);
    ML_ASSERT_EQ(parentScope.get(), scope->GetParent());

    parentScope->SetType("inParent", IRScalarValueType::Int8());
    parentScope->SetType("inBoth", IRScalarValueType::BFloat16());
    scope->SetType("inBoth", IRScalarValueType::String());
    scope->SetType("inChild", IRScalarValueType::Bool());

    ML_ASSERT_EQ(*IRScalarValueType::Int8(), *scope->GetType("inParent"));
    ML_ASSERT_EQ(*IRScalarValueType::Bool(), *scope->GetType("inChild"));
    ML_ASSERT_EQ(*IRScalarValueType::String(), *scope->GetType("inBoth"));

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

    parentScope->SetValue("inParent", IRScalarValueType::Bool()->Make(false));
    parentScope->SetValue("inBoth", IRScalarValueType::String()->Make<std::string>("parent"));
    scope->SetValue("inBoth", IRScalarValueType::String()->Make<std::string>("child"));
    scope->SetValue("inChild", IRScalarValueType::Float32()->Make(34.3f));

    ML_ASSERT_EQ(false, scope->GetValue("inParent")->AsBool());
    ML_ASSERT_EQ("child", scope->GetValue("inBoth")->AsString());
    ML_ASSERT_EQ(34.3f, scope->GetValue("inChild")->AsFloat32());

    ML_ASSERT_NULL(scope->TryGetType("undefined"));
    ML_ASSERT_THROWS(scope->GetType("undefined"), std::runtime_error);

    ML_ASSERT_THROWS(scope->GetValue("inParent", /*searchRoot=*/ false), std::runtime_error);
    ML_ASSERT_NULL(scope->TryGetValue("inParent", /*searchRoot=*/ false));

    return 0;
}
