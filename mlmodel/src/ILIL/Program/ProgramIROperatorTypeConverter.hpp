//
//  ProgramIROperatorTypeConverter.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IROperatorType.hpp"

#include <string>
#include <unordered_map>

namespace CoreML {
namespace ILIL {
namespace Program {

/**
 A converter from Spec V5 operator type to IROperatorType.
 */
class ProgramIROperatorTypeConverter {
public:
    ~ProgramIROperatorTypeConverter();
    ProgramIROperatorTypeConverter(const ProgramIROperatorTypeConverter&) = delete;
    ProgramIROperatorTypeConverter& operator=(const ProgramIROperatorTypeConverter&) = delete;
    ProgramIROperatorTypeConverter(ProgramIROperatorTypeConverter&&) = delete;
    ProgramIROperatorTypeConverter& operator=(ProgramIROperatorTypeConverter&&) = delete;

    /** Get the singleton instance. */
    static const ProgramIROperatorTypeConverter& Instance();

    IROperatorType GetType(const std::string& name) const;

private:
    ProgramIROperatorTypeConverter();

    using NameToTypeMap = std::unordered_map<std::string, IROperatorType>;

    static NameToTypeMap MakeNameToTypeMap();

    NameToTypeMap m_nameToType;
};

}
}
}
