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
    using IRFunctionMap = std::unordered_map<std::string, ConstIRFunctionPtr>;

    virtual ~IRProgram() = default;

    /** Get a specific function. */
    virtual const IRFunction& GetFunction(const std::string& name) const = 0;

    /** Get all declared functions. */
    virtual const IRFunctionMap& GetFunctions() const = 0;

    /** Get the value of the given parameter. */
    virtual const IRValue& GetParameterValue(const std::string& name) const = 0;

    /** Get the program's root scope. */
    virtual const IRScope& GetScope() const = 0;
};

}
}
