//
//  IROperatorType.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

namespace CoreML {
namespace ILIL {

/**
 The list of all operators.
 */
enum class IROperatorType {
    Activation,
    Const,
    Convolution,
    InnerProduct,
    Pooling,

    COUNT
};

}
}
