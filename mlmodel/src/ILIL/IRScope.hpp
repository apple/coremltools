//
//  IRScope.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRValue.hpp"

#include <unordered_map>

namespace CoreML {
namespace ILIL {

/**
 Lexical scoping implementation.

 Scopes represent a hierarchical collection of values. Constructs
 defining a scope in ILIL, including IRProgram and IRBlock, create
 and hold a shared pointer to one. For example, a simple program
 defining a single function with a single block will contain two
 scopes: a root scope containing program parameters and a nested
 scope holding the values of the function block.

 Values defined within a given ILIL construct are set in their
 corresponding scope. Queries for types and values starting in a
 given scope will continue through parents until found or the
 outermost scope is investigated.

 Every ILIL variable has a corresponding type entry in the scope
 created by the construct defining it, though only variables with
 an associated literal will have a value.
 */
class IRScope {
public:
    using ConstIRValueTypePtr = std::shared_ptr<const IRValueType>;
    using ConstIRValuePtr = std::shared_ptr<const IRValue>;

    ~IRScope();

    /** Create a new instance nested within the indicated scope. */
    IRScope(std::shared_ptr<const IRScope> parent);

    /**
     Get the type of the given value.
     If not found in this instance, the search will continue
     through parent scopes.
     @return a type pointer or nullptr if the type is not found.
     */
    ConstIRValueTypePtr GetType(const std::string& name) const;

    /** Associate the given type with the specified value name. */
    void SetType(const std::string& name, ConstIRValueTypePtr type);

    /**
     Get a value.
     If not found in this instance, the search will continue
     through parent scopes.
     @return a value pointer or nullptr if the value is not found.
     */
    ConstIRValuePtr GetValue(const std::string& name) const;

    /** Associate the given value with the specified value name. */
    void SetValue(const std::string& name, ConstIRValuePtr value);

private:
    std::shared_ptr<const IRScope> m_parent;
    std::unordered_map<std::string, ConstIRValueTypePtr> m_types;
    std::unordered_map<std::string, ConstIRValuePtr> m_values;
};

}
}
