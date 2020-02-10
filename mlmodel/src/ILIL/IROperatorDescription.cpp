//
//  IROperatorDescription.cpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/13/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IROperatorDescription.hpp"
#include "ILIL/IROperationValidator.hpp"
#include "Result.hpp"

#include <limits>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <vector>

using namespace CoreML::ILIL;

/** Checks if this valuetype matches another for the purposes of an operator's description.
    In particular, scalartypes and valuetypes are compared manually to verify
    equality. Type "any" is a wildcard for scalars and tensor shapes are not matched. */
static bool ValidateEquivalentTypes(const IRValueType& t1, const IRValueType& t2)
{
    auto t1_scalar = t1.TryAs<const IRScalarValueType>();
    if (t1_scalar) {
        auto t2_scalar = t2.TryAs<const IRScalarValueType>();
        if (!t2_scalar) return false;
        if (t2_scalar->GetType() == IRScalarValueTypeEnum::Any || t1_scalar->GetType() == IRScalarValueTypeEnum::Any) return true;
        return t2_scalar->GetType() == t1_scalar->GetType();
    }

    auto t1_tensor = t1.TryAs<const IRTensorValueType>();
    if (t1_tensor) {
        auto t2_tensor = t2.TryAs<const IRTensorValueType>();
        if (!t2_tensor) return false;
        // Only check if the size of both tensors are scalar (rank 0) or tensor (rank 1+).
        if ((t1_tensor->GetShape().size() == 0 && t2_tensor->GetShape().size() != 0) ||
            (t1_tensor->GetShape().size() != 0 && t2_tensor->GetShape().size() == 0)) {
            return 0;
        }
        
        return ValidateEquivalentTypes(t1_tensor->GetScalarType(), t2_tensor->GetScalarType());
    }

    auto t1_list = t1.TryAs<const IRListValueType>();
    if (t1_list) {
        auto t2_list = t2.TryAs<const IRListValueType>();
        if (!t2_list) return false;
        return ValidateEquivalentTypes(t1_list->GetElementType(), t2_list->GetElementType());
    }

    auto t1_tuple = t1.TryAs<const IRTupleValueType>();
    if (t1_tuple) {
        auto t2_tuple = t2.TryAs<const IRTupleValueType>();
        if (!t2_tuple) return false;
        if (t1_tuple->GetTypes().size() != t2_tuple->GetTypes().size()) {
            return false;
        }

        for (size_t i = 0; i < t1_tuple->GetTypes().size(); ++i) {
            if (!ValidateEquivalentTypes(*t1_tuple->GetTypes()[i], *t2_tuple->GetTypes()[i])) {
                return false;
            }
        }
    }
    return true;
}

IROperatorDescription::IROperatorDescription(uint64_t minOutputs,
                                             uint64_t maxOutputs,
                                             InputMapPtr expectedInputs,
                                             ValidationFunction validationFunction)
    : m_minOutputs(minOutputs)
    , m_maxOutputs(maxOutputs)
    , m_expectedInputs(std::move(expectedInputs))
    , m_validationFunction(validationFunction)
{ }

uint64_t IROperatorDescription::GetMinOutputs() const
{
    return m_minOutputs;
}

uint64_t IROperatorDescription::GetMaxOutputs() const
{
    return m_maxOutputs;
}

::CoreML::Result IROperatorDescription::ValidateOp(const IROperation& op) const
{
    return m_validationFunction(op);
}

const IROperatorDescription::InputMap& IROperatorDescription::GetExpectedInputs() const
{
    return *m_expectedInputs;
}

bool IROperatorDescription::IsValidType(const std::string& input, const IRValueType& type) const
{
    if (m_expectedInputs.get() == nullptr) return true;
    auto& validTypeList = m_expectedInputs->find(input)->second;
    for (auto t : *validTypeList) {
        if (ValidateEquivalentTypes(*t, type)) return true;
    }
    return false;
}

/* static */ IROperatorDescription::InputTypeSetPtr IROperatorDescription::MakeTypeList(std::initializer_list<IROperatorDescription::InputTypeSet::value_type> iolist) {
    return std::make_shared<IROperatorDescription::InputTypeSet>(iolist);
}

/* static */ IROperatorDescription::InputMapPtr IROperatorDescription::MakeInputMap(std::initializer_list<IROperatorDescription::InputMap::value_type> iolist) {
    return std::make_shared<IROperatorDescription::InputMap>(iolist);
}
