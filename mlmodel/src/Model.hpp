#ifndef ML_MODEL_HPP
#define ML_MODEL_HPP

#include <memory>
#include <sstream>
#include <string>

#include "Globals.hpp"
#include "Result.hpp"
#include "Validators.hpp"

#include "../build/format/Model_enums.h"
#include "../build/format/Normalizer_enums.h"

namespace CoreML {

namespace Specification {
    class Model;
    class ModelDescription;
    class Metadata;
}

namespace Model {
    void initModel(CoreML::Specification::Model* model, const std::string& description);
    void initClassifier(CoreML::Specification::Model* model, const std::string& predictedClassOutput, const std::string& predictedProbabilitiesOutput, const std::string& description);
    void initRegressor(CoreML::Specification::Model* model, const std::string& predictedValueOutput, const std::string& description);
    Result validateGeneric(const Specification::Model& model);

    /**
     * Deserializes a MLModel from an std::istream.
     *
     * @param stream the input stream.
     * @param out a MLModel reference that will be overwritten with the loaded
     *        MLModel if successful.
     * @return the result of the load operation, with ResultType::NO_ERROR on success.
     */
    Result load(CoreML::Specification::Model* out, std::istream& stream);
    
    /**
     * Deserializes a MLModel from a file given by a path.
     *
     * @param path the input file path.
     * @param out MLModel reference loaded MLModel if successful.
     * @return the result of the load operation, with ResultType::NO_ERROR on
     * success.
     */
    Result load(CoreML::Specification::Model* out, const std::string& path);

    /**
     * Serializes a MLModel to an std::ostream.
     *
     * @param stream the output stream.
     * @return the result of the save operation, with ResultType::NO_ERROR on success.
     */
    Result save(const CoreML::Specification::Model& model, std::ostream& stream);

    /**
     * Serializes a MLModel to a file given by a path.
     *
     * @param path the output file path.
     * @return result of save operation, with ResultType::NO_ERROR on success.
     */
    Result save(const CoreML::Specification::Model& model, const std::string& path);

    MLModelType modelType(const CoreML::Specification::Model& model);
    std::string modelTypeName(const CoreML::Specification::Model& model);

    SchemaType inputSchema(const CoreML::Specification::Model& model);
    SchemaType outputSchema(const CoreML::Specification::Model& model);

    /**
     * Enforces type invariant conditions.
     *
     * Enforces type invariant conditions for this TransformSpec, given a list
     * of allowed feature types and a proposed feature type. If the
     * proposed feature type is in the list of allowed feature types, this
     * function returns a Result with type ResultType::NO_ERROR. If the
     * proposed feature type is not in the list of allowed feature types,
     * this function returns a Result with type
     * ResultType::FEATURE_TYPE_INVARIANT_VIOLATION.
     *
     * @param allowedFeatureTypes the list of allowed feature types for
     *        this transform.
     * @param featureType the proposed feature type of a feature for this
     *        transform.
     * @return a Result corresponding to whether featureType is contained
     *         in allowedFeatureTypes.
     */
    Result enforceTypeInvariant(const std::vector<FeatureType>&
                                       allowedFeatureTypes, FeatureType featureType);
    
    
    /**
     * Ensures the spec is valid. This gets called every time before the
     * spec gets added to the MLModel.
     *
     * @return ResultType::NO_ERROR if the transform is valid.
     */
    Result validate(const Specification::Model& model);

    /**
     * Add an input to the transform-spec.
     *
     * @param featureName Name of the feature to be added.
     * @param featureType Type of the feature added.
     * @return Result type of this operation.
     */
    Result addInput(Specification::Model* model, const std::string& featureName, FeatureType featureType);
    
    /**
     * Add an output to the transform-spec.
     *
     * @param outputName  Name of the feature to be added.
     * @param outputType Type of the feature added.
     * @return Result type of this operation.
     */
    Result addOutput(Specification::Model* model, const std::string& outputName, FeatureType outputType);

    // string representation (text description)
    std::string toString(const CoreML::Specification::Model& model);
    void toStringStream(const CoreML::Specification::Model& model, std::stringstream& ss);

} // namespace Model

}

extern "C" {

/**
 * C-Structs needed for integration with pure C.
 */
typedef struct _MLModelSpecification {
    std::shared_ptr<CoreML::Specification::Model> cppFormat;
    _MLModelSpecification();
    _MLModelSpecification(const CoreML::Specification::Model&);
} MLModelSpecification;
    
typedef struct _MLModelMetadataSpecification {
    std::shared_ptr<CoreML::Specification::Metadata> cppMetadata;
    _MLModelMetadataSpecification();
    _MLModelMetadataSpecification(const CoreML::Specification::Metadata&);
} MLModelMetadataSpecification;

typedef struct _MLModelDescriptionSpecification {
    std::shared_ptr<CoreML::Specification::ModelDescription> cppInterface;
    _MLModelDescriptionSpecification();
    _MLModelDescriptionSpecification(const CoreML::Specification::ModelDescription&);
} MLModelDescriptionSpecification;
    
}
#endif
