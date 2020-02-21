//
//  IROperatorDescription.hpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/13/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRValueType.hpp"

#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <functional>

namespace CoreML {
class Result;
namespace ILIL {

class IROperation;

/**
A description of an ILIL operator input.
*/
class IROperatorInputDescription {
public:
    using InputTypeSet = std::unordered_set<std::shared_ptr<const IRValueType>>;
    using InputTypeSetPtr = std::shared_ptr<InputTypeSet>;

    IROperatorInputDescription() = delete;

    IROperatorInputDescription(bool isConst,
                               bool isOptional,
                               InputTypeSetPtr validTypeSet);
    IROperatorInputDescription(bool isConst,
                               InputTypeSetPtr validTypeSet);
    IROperatorInputDescription(InputTypeSetPtr validTypeSet);

    bool IsConst() const;
    bool IsOptional() const;
    bool IsValidType(const IRValueType& type) const;

    static InputTypeSetPtr MakeTypeList(std::initializer_list<InputTypeSet::value_type> iolist);

private:
    bool m_isConst;
    bool m_isOptional;
    InputTypeSetPtr m_validTypeSet;
}; // class IROperatorInputDescription

/**
 A description of an ILIL operator.
 */
class IROperatorDescription {
public:
    using InputMap = std::unordered_map<std::string, IROperatorInputDescription>;
    using InputMapPtr = std::shared_ptr<InputMap>;
    using ValidationFunction = std::function<Result(const IROperation&)>;
    
    IROperatorDescription() = delete;

    IROperatorDescription(uint64_t minOutputs,
                          uint64_t maxOutputs,
                          InputMapPtr expectedInputs,
                          ValidationFunction validationFunction);

    IROperatorDescription(uint64_t minOutputs,
                          uint64_t maxOutputs,
                          InputMapPtr expectedInputs);

    /** Get the minimum number of outputs this operator produces. */
    uint64_t GetMinOutputs() const;

    /** Get the maximum number of outputs this operator produces. */
    uint64_t GetMaxOutputs() const;

    /** Get list of inputs and their expected types. */
    const InputMap& GetExpectedInputs() const;

    /** Calls the validation function defined at class construction. */
    Result ValidateOp(const IROperation& op) const;

    static InputMapPtr MakeInputMap(std::initializer_list<IROperatorDescription::InputMap::value_type> iolist);
    
private:
    uint64_t m_minOutputs;
    uint64_t m_maxOutputs;
    InputMapPtr m_expectedInputs;
    ValidationFunction m_validationFunction;
}; // class IROperatorDescription

} // namespace ILIL
} // namespace CoreML
