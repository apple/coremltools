//
//  IRValueType.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include <string>
#include <vector>

namespace CoreML {
namespace ILIL {

/** The length of a dimension in a list, tensor, or tuple. */
class IRDimension {
public:
    virtual ~IRDimension();
    virtual bool operator==(const IRDimension& other) const = 0;
protected:
    IRDimension();
};

//-----------------------------------------------------------------

/** A dimension whose length is known at compile time. */
class IRConstantDimension : public IRDimension {
public:
    ~IRConstantDimension();
    IRConstantDimension(int64_t size);

    /** Get the length of this dimension. */
    int64_t GetSize() const;

    bool operator==(const IRDimension& other) const override;

private:
    int64_t m_size;
};

//-----------------------------------------------------------------

/** A named dimension whose value cannot be determined at compile time. */
class IRSymbolicDimension : public IRDimension {
public:
    ~IRSymbolicDimension();
    IRSymbolicDimension(const std::string& name);

    /** Get the name of this dimension. */
    const std::string& GetName() const;

    bool operator==(const IRDimension& other) const override;

private:
    std::string m_name;
};

//-----------------------------------------------------------------

class IRValueType {
public:
    virtual ~IRValueType();

    virtual bool operator==(const IRValueType& other) const = 0;
    bool operator!=(const IRValueType& other) const;

protected:
    IRValueType();
};

//-----------------------------------------------------------------

/** All scalar types. */
enum class IRScalarValueTypeEnum {
    Dynamic = 0,  // undefined / runtime type (e.g., trainable quantization)
    Bool = 1,
    String = 2,  // arbitrary sequence of bytes

    // Floats
    Float16 = 10,
    Float32 = 11,
    Float64 = 12,
    BFloat16 = 13,

    // Ints
    Int4 = 20,
    Int8 = 21,
    Int16 = 22,
    Int32 = 23,
    Int64 = 24,

    // UInts
    UInt4 = 30,
    UInt8 = 31,
    UInt16 = 32,
    UInt32 = 33,
    UInt64 = 34,
};

//-----------------------------------------------------------------

/** An IRValueType representing a scalar type. */
class IRScalarValueType : public IRValueType {
public:
    ~IRScalarValueType();

    IRScalarValueType(IRScalarValueTypeEnum type);

    IRScalarValueTypeEnum GetType() const;

    bool operator==(const IRValueType& other) const override;

private:
    IRScalarValueTypeEnum m_type;
};

//-----------------------------------------------------------------

/** An IRValueType representing a tensor type. */
class IRTensorValueType : public IRValueType {
public:
    using Shape = std::vector<std::shared_ptr<const IRDimension>>;

    ~IRTensorValueType();
    IRTensorValueType(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape);

    /** Get the type of element stored in this tensor type. */
    const IRScalarValueType& GetScalarType() const;

    /** Get the shape of this tensor type. */
    const Shape& GetShape() const;

    bool operator==(const IRValueType& other) const override;

private:
    std::shared_ptr<const IRScalarValueType> m_scalarType;
    Shape m_shape;
};

//-----------------------------------------------------------------

/** An IRValueType representing a list type. */
class IRListValueType : public IRValueType {
public:
    ~IRListValueType();
    IRListValueType(std::shared_ptr<const IRValueType> elementType,
                    std::shared_ptr<const IRDimension> length);

    /** Get the type of element stored in this list type. */
    const IRValueType& GetElementType() const;

    /** Get the length of lists of this type. */
    const IRDimension& GetLength() const;

    bool operator==(const IRValueType& other) const override;

private:
    std::shared_ptr<const IRValueType> m_elementType;
    std::shared_ptr<const IRDimension> m_length;
};

//-----------------------------------------------------------------

/** An IRValueType representing a tuple type. */
class IRTupleValueType : public IRValueType {
public:
    using ValueTypePtrVec = std::vector<std::shared_ptr<const IRValueType>>;

    ~IRTupleValueType();
    IRTupleValueType(ValueTypePtrVec&& types);

    /** Get the types of types in this tuple type. */
    const ValueTypePtrVec& GetTypes() const;

    bool operator==(const IRValueType& other) const override;

private:
    ValueTypePtrVec m_types;
};

//-----------------------------------------------------------------

/** A name/type pair. */
class IRNamedValueType {
public:
    ~IRNamedValueType();

    IRNamedValueType(const std::string& name,
                     std::shared_ptr<const IRValueType> type);

    const std::string& GetName() const;
    const IRValueType& GetType() const;

private:
    std::string m_name;
    std::shared_ptr<const IRValueType> m_type;
};

}
}
