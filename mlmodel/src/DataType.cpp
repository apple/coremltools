#include "Comparison.hpp"
#include "DataType.hpp"
#include "Format.hpp"

#include <sstream>

namespace CoreML {

    FeatureType::FeatureType(MLFeatureTypeType type)
    : m_type(std::make_shared<Specification::FeatureType>()) {
        switch (type) {
            case MLFeatureTypeType_NOT_SET:
                break;
            case MLFeatureTypeType_multiArrayType:
                m_type->mutable_multiarraytype();
                break;
            case MLFeatureTypeType_imageType:
                m_type->mutable_imagetype();
                break;
            case MLFeatureTypeType_int64Type:
                m_type->mutable_int64type();
                break;
            case MLFeatureTypeType_doubleType:
                m_type->mutable_doubletype();
                break;
            case MLFeatureTypeType_stringType:
                m_type->mutable_stringtype();
                break;
            case MLFeatureTypeType_dictionaryType:
                m_type->mutable_dictionarytype();
                break;
        }
    }

    FeatureType::FeatureType(const Specification::FeatureType& wrapped)
    : m_type(std::make_shared<Specification::FeatureType>(wrapped)) {
    }
    
    // simple types
#define WRAP_SIMPLE_TYPE(T, U) \
FeatureType FeatureType::T() { return FeatureType(U); }
    
    WRAP_SIMPLE_TYPE(Int64, MLFeatureTypeType_int64Type)
    WRAP_SIMPLE_TYPE(String, MLFeatureTypeType_stringType)
    WRAP_SIMPLE_TYPE(Image, MLFeatureTypeType_imageType) /* TODO image is not simple type */
    WRAP_SIMPLE_TYPE(Double, MLFeatureTypeType_doubleType)
    
    // parametric types
    FeatureType FeatureType::Array(const std::vector<int64_t> shape, MLArrayDataType dataType) {
        FeatureType out(MLFeatureTypeType_multiArrayType);
        Specification::ArrayFeatureType *params = out->mutable_multiarraytype();
        
        for (int64_t s : shape) {
            params->add_shape(s);
        }
        params->set_datatype(static_cast<Specification::ArrayFeatureType::ArrayDataType>(dataType));
        return out;
    }

    FeatureType FeatureType::Array(const std::vector<int64_t> shape) {
        return Array(shape,MLArrayDataTypeDOUBLE);
    }
    
    FeatureType FeatureType::Dictionary(MLDictionaryFeatureTypeKeyType keyType) {
        FeatureType out(MLFeatureTypeType_dictionaryType);

        Specification::DictionaryFeatureType *params = out->mutable_dictionarytype();
        
        switch (keyType) {
            case MLDictionaryFeatureTypeKeyType_int64KeyType:
                params->mutable_int64keytype();
                break;
            case MLDictionaryFeatureTypeKeyType_stringKeyType:
                params->mutable_stringkeytype();
                break;
            case MLDictionaryFeatureTypeKeyType_NOT_SET:
                throw std::runtime_error("Invalid dictionary key type. Expected one of: {int64, string}.");
        }

        return out;
    }
    
    // operators
    const Specification::FeatureType& FeatureType::operator*() const {
        return *m_type;
    }
    
    Specification::FeatureType& FeatureType::operator*() {
        return *m_type;
    }
    
    const Specification::FeatureType* FeatureType::operator->() const {
        return m_type.get();
    }
    
    Specification::FeatureType* FeatureType::operator->() {
        return m_type.get();
    }
    
    bool FeatureType::operator==(const FeatureType& other) const {
        return *m_type == *other.m_type;
    }
    
    bool FeatureType::operator!=(const FeatureType& other) const {
        return !(*this == other);
    }

    static std::string featureTypeToString(Specification::FeatureType::TypeCase tag) {
        switch (tag) {
            case Specification::FeatureType::kMultiArrayType:
                return "MultiArray";
            case Specification::FeatureType::kDictionaryType:
                return "Dictionary";
            case Specification::FeatureType::kImageType:
                return "Image";
            case Specification::FeatureType::kDoubleType:
                return "Double";
            case Specification::FeatureType::kInt64Type:
                return "Int64";
            case Specification::FeatureType::kStringType:
                return "String";
            case Specification::FeatureType::TYPE_NOT_SET:
                return "Invalid";
        }
    }
    
    static std::string keyTypeToString(Specification::DictionaryFeatureType::KeyTypeCase tag) {
        switch (tag) {
            case Specification::DictionaryFeatureType::kInt64KeyType:
                return "Int64";
            case Specification::DictionaryFeatureType::kStringKeyType:
                return "String";
            case Specification::DictionaryFeatureType::KEYTYPE_NOT_SET:
                return "Invalid";
        }
    }

    static std::string dataTypeToString(Specification::ArrayFeatureType_ArrayDataType dataType) {
        switch (dataType) {
            case Specification::ArrayFeatureType_ArrayDataType_INT32:
                return "Int32";
            case Specification::ArrayFeatureType_ArrayDataType_DOUBLE:
                return "Double";
            case Specification::ArrayFeatureType_ArrayDataType_FLOAT32:
                return "Float32";
            case Specification::ArrayFeatureType_ArrayDataType_INVALID_ARRAY_DATA_TYPE:
            case Specification::ArrayFeatureType_ArrayDataType_ArrayFeatureType_ArrayDataType_INT_MAX_SENTINEL_DO_NOT_USE_:
            case Specification::ArrayFeatureType_ArrayDataType_ArrayFeatureType_ArrayDataType_INT_MIN_SENTINEL_DO_NOT_USE_:
                return "Invalid";
        }
    }

    static std::string colorSpaceToString(Specification::ImageFeatureType_ColorSpace colorspace) {
        switch (colorspace) {
            case Specification::ImageFeatureType_ColorSpace_BGR:
                return "BGR";
            case Specification::ImageFeatureType_ColorSpace_RGB:
                return "RGB";
            case Specification::ImageFeatureType_ColorSpace_GRAYSCALE:
                return "Grayscale";
            case Specification::ImageFeatureType_ColorSpace_ImageFeatureType_ColorSpace_INT_MAX_SENTINEL_DO_NOT_USE_:
            case Specification::ImageFeatureType_ColorSpace_ImageFeatureType_ColorSpace_INT_MIN_SENTINEL_DO_NOT_USE_:
            case Specification::ImageFeatureType_ColorSpace_INVALID_COLOR_SPACE:
                return "Invalid";
        }
    }

    // methods
    std::string FeatureType::toString() const {
        std::stringstream ss;
        Specification::FeatureType::TypeCase tag = m_type->Type_case();

        ss << featureTypeToString(tag);

        switch (tag) {
            case Specification::FeatureType::kMultiArrayType:
            {
                const Specification::ArrayFeatureType& params = m_type->multiarraytype();
                ss << " (" << dataTypeToString(params.datatype());
                int shapeSize = params.shape_size();
                if (shapeSize > 0) { ss << " "; }
                for (int i=0; i<shapeSize; i++) {
                    ss << params.shape(i);
                    if (i < shapeSize - 1) { ss << " x "; }
                }
                ss << ")";
                break;
            }
            case Specification::FeatureType::kDictionaryType:
            {
                const Specification::DictionaryFeatureType& params = m_type->dictionarytype();
                ss << " (";
                ss << keyTypeToString(params.KeyType_case());
                ss << " → ";
                ss << featureTypeToString(Specification::FeatureType::kDoubleType); // assume double value
                ss << ")";
                break;
            }
            case Specification::FeatureType::kImageType:
            {
                const Specification::ImageFeatureType& params = m_type->imagetype();
                ss << " (";
                ss << colorSpaceToString(params.colorspace());
                ss << " " << params.width() << " x " << params.height();
                ss << ")";
                break;
            }
            default:
                break;
        }
        return ss.str() + (m_type->isoptional() ? "?" : "");
    }

    std::map<std::string,std::string> FeatureType::toDictionary() const {
        Specification::FeatureType::TypeCase tag = m_type->Type_case();

        std::map<std::string, std::string> dict;
        dict["type"] = featureTypeToString(tag);
        dict["isOptional"] = m_type->isoptional() ? "1" : "0";

        switch (tag) {
            case Specification::FeatureType::kMultiArrayType:
            {
                const Specification::ArrayFeatureType& params = m_type->multiarraytype();

                dict["dataType"] = dataTypeToString(params.datatype());

                int shapeSize = params.shape_size();
                if (shapeSize > 0) {
                    std::stringstream ss;
                    ss << "[";
                    for (int i=0; i<shapeSize; i++) {
                        ss << params.shape(i);
                        if (i < shapeSize - 1) { ss << ", "; }
                    }
                    ss << "]";
                    dict["shape"] = ss.str();
                }
                break;
            }
            case Specification::FeatureType::kDictionaryType:
            {
                dict["keyType"] = keyTypeToString(m_type->dictionarytype().KeyType_case());
                break;
            }
            case Specification::FeatureType::kImageType:
            {
                const Specification::ImageFeatureType& params = m_type->imagetype();
                dict["width"] = std::to_string(params.width());
                dict["height"] = std::to_string(params.height());
                dict["colorspace"] = colorSpaceToString(params.colorspace());
                dict["isColor"] = params.colorspace() == Specification::ImageFeatureType_ColorSpace_GRAYSCALE ? "0" : "1";
                break;
            }
            default:
                break;
        }

        return dict;
    }
    
    Specification::FeatureType* FeatureType::allocateCopy() {
        // we call new here, but don't free!
        // this method should only be called immediately prior to passing the
        // returned pointer into a protobuf method that expects to take ownership
        // over the heap object pointed to.
        return new Specification::FeatureType(*m_type);
    }

}
