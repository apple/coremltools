//
//  IRValueType.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "IRValue.hpp"

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
    bool operator!=(const IRDimension& other) const;

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

    /** Is this an IRDimension of the given type? */
    template<typename DimensionT>
    bool Is() const {
        return TryAs<DimensionT>() != nullptr;
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

    /** Convenience factory method. */
    static std::unique_ptr<IRConstantDimension> Make(uint64_t size);

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

    /** Convenience factory method. */
    static std::unique_ptr<IRSymbolicDimension> Make(const std::string& name);

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

    IRValueType(const IRValueType&) = delete;
    IRValueType(IRValueType&&) = delete;
    IRValueType& operator=(const IRValueType&) = delete;
    IRValueType& operator=(IRValueType&&) = delete;

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

    /** Is this an IRValueType of the given type? */
    template<typename ValueTypeT>
    bool Is() const {
        return TryAs<ValueTypeT>() != nullptr;
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
    
    // Any Scalar Value
    // Should only be used in the context of validating inputs.
    Any = 99,
};

//-----------------------------------------------------------------

/** An IRValueType representing a scalar type. */
class IRScalarValueType : public IRValueType, public std::enable_shared_from_this<IRScalarValueType> {
public:
    ~IRScalarValueType();

    IRScalarValueTypeEnum GetType() const;

    /** Make a new immediate scalar value. */
    template<typename ScalarT>
    std::unique_ptr<const IRScalarValue<ScalarT>> Make(ScalarT value) const;

    uint64_t GetNumElements() const override;
    std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
    bool operator==(const IRValueType& other) const override;

    // ---------------------------------------------
    // Convenience methods to get IRScalarValueTypes
    static std::shared_ptr<const IRScalarValueType> Dynamic();
    static std::shared_ptr<const IRScalarValueType> Bool();
    static std::shared_ptr<const IRScalarValueType> String();
    static std::shared_ptr<const IRScalarValueType> Float16();
    static std::shared_ptr<const IRScalarValueType> Float32();
    static std::shared_ptr<const IRScalarValueType> Float64();
    static std::shared_ptr<const IRScalarValueType> BFloat16();
    static std::shared_ptr<const IRScalarValueType> Int4();
    static std::shared_ptr<const IRScalarValueType> Int8();
    static std::shared_ptr<const IRScalarValueType> Int16();
    static std::shared_ptr<const IRScalarValueType> Int32();
    static std::shared_ptr<const IRScalarValueType> Int64();
    static std::shared_ptr<const IRScalarValueType> UInt4();
    static std::shared_ptr<const IRScalarValueType> UInt8();
    static std::shared_ptr<const IRScalarValueType> UInt16();
    static std::shared_ptr<const IRScalarValueType> UInt32();
    static std::shared_ptr<const IRScalarValueType> UInt64();
    static std::shared_ptr<const IRScalarValueType> Any();
private:
    IRScalarValueType(IRScalarValueTypeEnum type);

    IRScalarValueTypeEnum m_type;
};

//-----------------------------------------------------------------

/** An IRValueType representing a tensor type. */
class IRTensorValueType : public IRValueType, public std::enable_shared_from_this<IRTensorValueType> {
public:
    using Shape = std::vector<std::shared_ptr<const IRDimension>>;

    ~IRTensorValueType();

    /** Create a new instance. */
    static std::shared_ptr<const IRTensorValueType>
    Make(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape);

    /** Create a new instance with no shape information */
    static std::shared_ptr<const IRTensorValueType>
    Make(std::shared_ptr<const IRScalarValueType> scalarType);

    /** Get the type of element stored in this tensor type. */
    const IRScalarValueType& GetScalarType() const;

    /** Get the shape of this tensor type. */
    const Shape& GetShape() const;

    /** Make a new immediate tensor value. */
    template<typename ScalarT>
    std::unique_ptr<const IRTensorValue<ScalarT>> Make(std::vector<ScalarT>&& values) const;

    uint64_t GetNumElements() const override;
        std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
    bool operator==(const IRValueType& other) const override;

private:
    IRTensorValueType(std::shared_ptr<const IRScalarValueType> scalarType, Shape&& shape);

    std::shared_ptr<const IRScalarValueType> m_scalarType;
    Shape m_shape;
};

//-----------------------------------------------------------------

/** An IRValueType representing a list type. */
class IRListValueType : public IRValueType, public std::enable_shared_from_this<IRListValueType> {
public:
    ~IRListValueType();

    /** Create a new instance. */
    static std::shared_ptr<const IRListValueType> Make(std::shared_ptr<const IRValueType> elementType,
                                                       std::shared_ptr<const IRDimension> length);

    /** Get the type of element stored in this list type. */
    const IRValueType& GetElementType() const;

    /** Get the length of lists of this type. */
    const IRDimension& GetLength() const;

    uint64_t GetNumElements() const override;
    std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
    bool operator==(const IRValueType& other) const override;

private:
    IRListValueType(std::shared_ptr<const IRValueType> elementType,
                    std::shared_ptr<const IRDimension> length);

    std::shared_ptr<const IRValueType> m_elementType;
    std::shared_ptr<const IRDimension> m_length;
};

//-----------------------------------------------------------------

/** An IRValueType representing a tuple type. */
class IRTupleValueType : public IRValueType, public std::enable_shared_from_this<IRTupleValueType> {
public:
    using ConstIRValueVec = IRTupleValue::ConstIRValueVec;
    using ValueTypePtrVec = std::vector<std::shared_ptr<const IRValueType>>;

    ~IRTupleValueType();

    /** Create a new instance. */
    static std::shared_ptr<const IRTupleValueType> Make(ValueTypePtrVec&& types);

    /** Get the types of types in this tuple type. */
    const ValueTypePtrVec& GetTypes() const;

    /** Make a new immediate tuple value. */
    std::unique_ptr<const IRTupleValue> Make(ConstIRValueVec&& values) const;

    uint64_t GetNumElements() const override;
    std::unique_ptr<const IRValue> ReadValue(const std::string& filePath, uint64_t offset) const override;
    bool operator==(const IRValueType& other) const override;

private:
    IRTupleValueType(ValueTypePtrVec&& types);

    ValueTypePtrVec m_types;
};

//-----------------------------------------------------------------

/** A name/type pair. */
class IRNamedValueType {
public:
    ~IRNamedValueType();

    /** Create a new instance. */
    static std::shared_ptr<const IRNamedValueType> Make(const std::string& name,
                                                        std::shared_ptr<const IRValueType> type);

    const std::string& GetName() const;
    const IRValueType& GetType() const;

private:
    IRNamedValueType(const std::string& name,
                     std::shared_ptr<const IRValueType> type);

    std::string m_name;
    std::shared_ptr<const IRValueType> m_type;
};

}
}
