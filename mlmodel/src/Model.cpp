#include "Comparison.hpp"
#include "Format.hpp"
#include "Model.hpp"
#include "Utils.hpp"

#include <fstream>
#include <unordered_map>

namespace CoreML {
namespace Model {
    void initRegressor(CoreML::Specification::Model* model, const std::string& predictedValueOutput, const std::string& description) {
        model->set_specificationversion(MLMODEL_SPECIFICATION_VERSION);
        model->mutable_description()->set_predictedfeaturename(predictedValueOutput);
        Specification::Metadata* metadata = model->mutable_description()->mutable_metadata();
        metadata->set_shortdescription(description);
    }

    Result validateGeneric(const Specification::Model& model) {
        // make sure compat version fields are filled in
        if (model.specificationversion() == 0) {
            return Result(ResultType::INVALID_COMPATIBILITY_VERSION,
                          "Model specification version field missing or corrupt.");
        }

        // check public compatibility version
        // note: this one should always be backward compatible, so use > here
        if (model.specificationversion() > MLMODEL_SPECIFICATION_VERSION) {
            std::stringstream msg;
            msg << "The .mlmodel supplied is of version "
                << model.specificationversion()
                << ", intended for a newer version of Xcode. This version of Xcode supports model version "
                << MLMODEL_SPECIFICATION_VERSION
                << " or earlier.";
            return Result(ResultType::UNSUPPORTED_COMPATIBILITY_VERSION,
                          msg.str());
        }

        // validate model interface
        Result r = validateModelDescription(model.description(), model.specificationversion());
        if (!r.good()) {
            return r;
        }

        return validateOptional(model);
    }

#define VALIDATE_MODEL_TYPE(TYPE) \
    case MLModelType_ ## TYPE: \
        return ::CoreML::validate<MLModelType_ ## TYPE>(model);

    Result validate(const Specification::Model& model) {
        Result result = validateGeneric(model);
        if (!result.good()) { return result; }

        MLModelType type = static_cast<MLModelType>(model.Type_case());
        // TODO -- is there a better way to do this than switch/case?
        // the compiler doesn't like <type> being unknown at compile time.
        switch (type) {
                VALIDATE_MODEL_TYPE(pipelineClassifier);
                VALIDATE_MODEL_TYPE(pipelineRegressor);
                VALIDATE_MODEL_TYPE(pipeline);
                VALIDATE_MODEL_TYPE(glmClassifier);
                VALIDATE_MODEL_TYPE(glmRegressor);
                VALIDATE_MODEL_TYPE(treeEnsembleClassifier);
                VALIDATE_MODEL_TYPE(treeEnsembleRegressor);
                VALIDATE_MODEL_TYPE(supportVectorClassifier);
                VALIDATE_MODEL_TYPE(supportVectorRegressor);
                VALIDATE_MODEL_TYPE(neuralNetworkClassifier);
                VALIDATE_MODEL_TYPE(neuralNetworkRegressor);
                VALIDATE_MODEL_TYPE(neuralNetwork);
                VALIDATE_MODEL_TYPE(oneHotEncoder);
                VALIDATE_MODEL_TYPE(arrayFeatureExtractor);
                VALIDATE_MODEL_TYPE(featureVectorizer);
                VALIDATE_MODEL_TYPE(imputer);
                VALIDATE_MODEL_TYPE(dictVectorizer);
                VALIDATE_MODEL_TYPE(scaler);
                VALIDATE_MODEL_TYPE(nonMaximumSuppression);
                VALIDATE_MODEL_TYPE(categoricalMapping);
                VALIDATE_MODEL_TYPE(normalizer);
                VALIDATE_MODEL_TYPE(identity);
                VALIDATE_MODEL_TYPE(customModel);
                VALIDATE_MODEL_TYPE(bayesianProbitRegressor);
                VALIDATE_MODEL_TYPE(wordTagger);
                VALIDATE_MODEL_TYPE(textClassifier);
                VALIDATE_MODEL_TYPE(visionFeaturePrint);
            case MLModelType_NOT_SET:
                return Result(ResultType::INVALID_MODEL_INTERFACE, "Model did not specify a valid model-parameter type.");
        }
    }

    Result load(Specification::Model* out, std::istream& in) {
        if (!in.good()) {
            return Result(ResultType::UNABLE_TO_OPEN_FILE,
                          "unable to open file for read");
        }

        
        Result r = loadSpecification(*out, in);
        if (!r.good()) { return r; }
        // validate on load
        r = validate(*out);

        return r;
    }
    
    Result load(Specification::Model* out, const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return load(out, in);
    }
    
    
    Result save(const Specification::Model& to_copy, std::ostream& out) {
        if (!out.good()) {
            return Result(ResultType::UNABLE_TO_OPEN_FILE,
                          "unable to open file for write");
        }

        Specification::Model model = to_copy;
        CoreML::downgradeSpecificationVersion(&model);

        // validate on save
        Result r = validate(model);
        if (!r.good()) {
            return r;
        }
        
        return saveSpecification(model, out);
    }
    
    Result save(const Specification::Model& model, const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        return save(model, out);
    }

    SchemaType inputSchema(const Specification::Model& model) {
        SchemaType inputs;
        const Specification::ModelDescription& interface = model.description();
        int size = interface.input_size();
        assert(size >= 0);
        inputs.reserve(static_cast<size_t>(size));
        for (int i = 0; i < size; i++) {
            const Specification::FeatureDescription &desc = interface.input(i);
            inputs.push_back(std::make_pair(desc.name(), desc.type()));
        }
        return inputs;
    }
    
    SchemaType outputSchema(const Specification::Model& model) {
        SchemaType outputs;
        const Specification::ModelDescription& interface = model.description();
        int size = interface.output_size();
        assert(size >= 0);
        outputs.reserve(static_cast<size_t>(size));
        for (int i = 0; i < size; i++) {
            const Specification::FeatureDescription &desc = interface.output(i);
            outputs.push_back(std::make_pair(desc.name(), desc.type()));
        }
        return outputs;
    }
    
    Result addInput(Specification::Model* model,
                    const std::string& featureName,
                    FeatureType featureType) {
        Specification::ModelDescription* interface = model->mutable_description();
        Specification::FeatureDescription *arg = interface->add_input();
        arg->set_name(featureName);
        arg->set_allocated_type(featureType.allocateCopy());
        return Result();
    }
    
    Result addOutput(Specification::Model* model,
                     const std::string& targetName,
                     FeatureType targetType) {
        Specification::ModelDescription* interface = model->mutable_description();
        Specification::FeatureDescription *arg = interface->add_output();
        arg->set_name(targetName);
        arg->set_allocated_type(targetType.allocateCopy());
        return Result();
    }
    
    MLModelType modelType(const Specification::Model& model) {
        return static_cast<MLModelType>(model.Type_case());
    }

    std::string modelTypeName(const Specification::Model& model) {
        return MLModelType_Name(modelType(model));
    }
    
    Result enforceTypeInvariant(const std::vector<FeatureType>& allowedFeatureTypes,
                                       FeatureType featureType) {
        
        for (const FeatureType& t : allowedFeatureTypes) {
            if (featureType == t) {
                // no invariant broken -- type matches one of the allowed types
                return Result();
            }
        }
        
        return Result::featureTypeInvariantError(allowedFeatureTypes, featureType);
    }
    
    static void writeFeatureDescription(std::stringstream& ss,
                                        const Specification::FeatureDescription& feature) {
        ss  << "\t\t"
            << feature.name()
            << " ("
            << FeatureType(feature.type()).toString()
            << ")";
        if (feature.shortdescription() != "") {
            ss << ": " << feature.shortdescription();
        }
        ss << "\n";
    }
    
    void toStringStream(const Specification::Model& model, std::stringstream& ss) {
        ss << "Spec version: " << model.specificationversion() << "\n";
        ss << "Model type: " << MLModelType_Name(static_cast<MLModelType>(model.Type_case())) << "\n";
        ss << "Interface:" << "\n";
        ss << "\t" << "Inputs:" << "\n";
        for (const auto& input : model.description().input()) {
            writeFeatureDescription(ss, input);
        }
        ss << "\t" << "Outputs:" << "\n";
        for (const auto& output : model.description().output()) {
            writeFeatureDescription(ss, output);
        }
        if (model.description().predictedfeaturename() != "") {
            ss << "\t" << "Predicted feature name: " << model.description().predictedfeaturename() << "\n";
        }
        if (model.description().predictedprobabilitiesname() != "") {
            ss << "\t" << "Predicted probability name: " << model.description().predictedprobabilitiesname() << "\n";
        }
    }
    
    std::string toString(const Specification::Model& model) {
        std::stringstream ss;
        toStringStream(model, ss);
        return ss.str();
    }

} // namespace Model
} // namespace CoreML

#pragma mark C exports

extern "C" {

_MLModelSpecification::_MLModelSpecification()
    : cppFormat(new CoreML::Specification::Model())
{
}

    _MLModelSpecification::_MLModelSpecification(const CoreML::Specification::Model& te)
: cppFormat(new CoreML::Specification::Model(te))
{
}

_MLModelMetadataSpecification::_MLModelMetadataSpecification() : cppMetadata(new CoreML::Specification::Metadata())
{
}
    
_MLModelMetadataSpecification::_MLModelMetadataSpecification(const CoreML::Specification::Metadata& meta)
: cppMetadata(new CoreML::Specification::Metadata(meta))
{
}
    
_MLModelDescriptionSpecification::_MLModelDescriptionSpecification() : cppInterface(new CoreML::Specification::ModelDescription())
{
}

_MLModelDescriptionSpecification::_MLModelDescriptionSpecification(const CoreML::Specification::ModelDescription& interface)
: cppInterface(new CoreML::Specification::ModelDescription(interface))
{
}

} // extern "C"
