//
//  IRProgram.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRFunction.hpp"

namespace CoreML {
namespace ILIL {

/** An ILIL program. */
class IRProgram {
public:
    using ConstIRFunctionPtr = std::shared_ptr<const IRFunction>;
    using ConstIRValuePtr = std::shared_ptr<const IRValue>;
    using IRFunctionMap = std::unordered_map<std::string, ConstIRFunctionPtr>;
    using ParameterMap = std::unordered_map<std::string, ConstIRValuePtr>;

    virtual ~IRProgram() = default;

    /**
     Get a specific function.
     @throw std::out_of_range if the function does not exist.
     */
    virtual const IRFunction& GetFunction(const std::string& name) const = 0;

    /** Get all declared functions. */
    virtual const IRFunctionMap& GetFunctions() const = 0;

    /** Get the names and values of all parameters. */
    virtual const ParameterMap GetParameters() const = 0;

    /** Get the names of all parameters. */
    virtual const std::vector<std::string>& GetParameterNames() const = 0;

    /**
     Get the value of the given parameter.
     @throw std::out_of_range if the parameter does not exist.
     */
    virtual const IRValue& GetParameterValue(const std::string& name) const = 0;

    /**
     Get the value of the given parameter.
     @return The parameter value or nullptr if it does not exist.
     */
    virtual ConstIRValuePtr TryGetParameterValue(const std::string& name) const = 0;

    /** Get the program's root scope. */
    virtual const IRScope& GetScope() const = 0;
};

}
}
