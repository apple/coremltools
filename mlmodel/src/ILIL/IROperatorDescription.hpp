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
 A description of an ILIL operator.
 */
class IROperatorDescription {
public:
    using InputTypeSet = std::unordered_set<std::shared_ptr<const IRValueType>>;
    using InputTypeSetPtr = std::shared_ptr<InputTypeSet>;
    using InputMap = std::unordered_map<std::string, InputTypeSetPtr>;
    using InputMapPtr = std::shared_ptr<InputMap>;
    using ValidationFunction = std::function<Result(const IROperation&)>;

    IROperatorDescription(uint64_t minOutputs,
                          uint64_t maxOutputs,
                          InputMapPtr expectedInputs,
                          ValidationFunction validate);

    /** Get the minimum number of outputs this operator produces. */
    uint64_t GetMinOutputs() const;

    /** Get the maximum number of outputs this operator produces. */
    uint64_t GetMaxOutputs() const;

    /** Get list of inputs and their expected types */
    const InputMap& GetExpectedInputs() const;

    Result ValidateOp(const IROperation& op) const;

    bool IsValidType(const std::string& input, const IRValueType& type) const;
    static InputTypeSetPtr MakeTypeList(std::initializer_list<IROperatorDescription::InputTypeSet::value_type> iolist);
    static InputMapPtr MakeInputMap(std::initializer_list<IROperatorDescription::InputMap::value_type> iolist);
    
private:
    uint64_t m_minOutputs;
    uint64_t m_maxOutputs;
    InputMapPtr m_expectedInputs;
    ValidationFunction m_validationFunction;
}; // class IROperatorDescription

} // namespace ILIL
} // namespace CoreML
