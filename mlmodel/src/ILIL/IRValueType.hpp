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

class IRValue;

/** The length of a dimension in a list, tensor, or tuple. */
class IRDimension {
public:
    virtual ~IRDimension();
    virtual bool operator==(const IRDimension& other) const = 0;

    /**
     Attempt to cast this instance to a more specific IRDimension.
     @throws std::bad_cast on failure.
     */
    template<typename DimensionT>
    const DimensionT* As() const {
        auto as = TryAs<DimensionT>();
        if (as) {
            return as;
        }
        throw std::bad_cast();
    }

    /**
     Attempt to cast this instance to a more specific IRDimension.
     @returns A pointer to DimensionT or nullptr on failure.
     */
    template<typename DimensionT>
    const DimensionT* TryAs() const {
        return dynamic_cast<const DimensionT*>(this);
    }

protected:
    IRDimension();
};

//-----------------------------------------------------------------

/** A dimension whose length is known at compile time. */
class IRConstantDimension : public IRDimension {
public:
    ~IRConstantDimension();
    IRConstantDimension(uint64_t size);

    /** Get the length of this dimension. */
    uint64_t GetSize() const;

    bool operator==(const IRDimension& other) const override;

private:
    uint64_t m_size;
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

    /**
     How many individual elements are held in a value of this type?

     Throws std::range_error if the type contains symbolic lengths/dimensions.
     */
    virtual uint64_t GetNumElements() const = 0;

    /**
     Attempt to cast this instance to a more specific IRValueType.
     @throws std::bad_cast on failure.
     */
    template<typename ValueTypeT>
    const ValueTypeT* As() const {
        auto as = TryAs<ValueTypeT>();
        if (as) {
            return as;
        }
        throw std::bad_cast();
    }

    /**
     Attempt to cast this instance to a more specific IRValueType.
     @returns A pointer to ValueTypeT or nullptr on failure.
     */
    template<typename ValueTypeT>
    const ValueTypeT* TryAs() const {
        return dynamic_cast<const ValueTypeT*>(this);
    }

    /** Read a value from the named file. */
    virtual std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const = 0;

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

    uint64_t GetNumElements() const override;
    std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
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

    uint64_t GetNumElements() const override;
        std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
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

    uint64_t GetNumElements() const override;
    std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
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

    uint64_t GetNumElements() const override;
    std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
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
