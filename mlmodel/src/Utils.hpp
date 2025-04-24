#ifndef MLMODEL_UTILS
#define MLMODEL_UTILS

#include <fstream>
#include <memory>
#include <string>
#include <sstream>
#include <limits>
#include <unordered_map>
#include <vector>
#include <functional>

#include "Globals.hpp"
#include "Model.hpp"
#include "Result.hpp"
#include "Format.hpp"

namespace CoreML {

    // This is the type used internally
    typedef unsigned short float16;

    template <typename T>
    static inline Result saveSpecification(const T& formatObj,
                                           std::ostream& out) {
        google::protobuf::io::OstreamOutputStream rawOutput(&out);

        if (!formatObj.SerializeToZeroCopyStream(&rawOutput)) {
            return Result(ResultType::FAILED_TO_SERIALIZE,
                          "unable to serialize object");
        }

        return Result();
    }


    static inline Result saveSpecificationPath(const Specification::Model& formatObj,
                                               const std::string& path) {
        Model m(formatObj);
        return m.save(path);
    }

    template <typename T>
    static inline Result loadSpecification(T& formatObj,
                                           std::istream& in) {

        google::protobuf::io::IstreamInputStream rawInput(&in);
        google::protobuf::io::CodedInputStream codedInput(&rawInput);

        // Support models up to 2GB
        codedInput.SetTotalBytesLimit(std::numeric_limits<int>::max());

        if (!formatObj.ParseFromCodedStream(&codedInput)) {
            return Result(ResultType::FAILED_TO_DESERIALIZE,
                          "unable to deserialize object");
        }

        return Result();
    }

    static inline Result loadSpecificationPath(Specification::Model& formatObj,
                                               const std::string& path) {
        Model m;
        Result r = CoreML::Model::load(path, m);
        if (!r.good()) { return r; }
        formatObj = m.getProto();
        return Result();
    }

    /**
     * If a model spec does not use features from later specification versions, this will
     * set the spec version so that the model can be executed on older versions of
     * Core ML. It applies recursively to sub models
     */
    void downgradeSpecificationVersion(Specification::Model *pModel);

    /**
     * Returns true if the specified model is a neural network and it has a custom layer.
     */
    bool hasCustomLayer(const Specification::Model& model);

    /**
     * The method "hasIOSXFeatures" will return false if the model does not express any "feature"
     * that was added in iOS X or later. When it returns false, the
     * "downgradeSpecificationVersion" function will downgrade the spec version from
     * MLMODEL_SPECIFICATION_VERSION_IOS_X to MLMODEL_SPECIFICATION_VERSION_IOS_X-1.
     * TODO: rdar://76017556
     * Right now to determine whether a new "feature" is present or not in the model spec,
     * presence of certain proto messages are checked. However, in future we might have a scenario where
     * no proto message is changed, but the spec contains a new feature (e.g.: a new op in the milProgram opset).
     * It is still to be determined (tracked by the radar above), what the course of action should be, whether the
     * spec version just versions the proto messages or the runtime feasibility of the contents of the proto.
     */
    bool hasIOS11_2Features(const Specification::Model& model);
    bool hasIOS12Features(const Specification::Model& model);
    bool hasIOS13Features(const Specification::Model& model);
    bool hasIOS14Features(const Specification::Model& model);
    bool hasIOS15Features(const Specification::Model& model);
    bool hasIOS16Features(const Specification::Model& model);
    bool hasIOS17Features(const Specification::Model& model);
    bool hasIOS18Features(const Specification::Model& model);

    typedef std::pair<std::string,std::string> StringPair;
    // Returns a vector of pairs of strings, one pair per custom layer instance
    std::vector<StringPair> getCustomLayerNamesAndDescriptions(const Specification::Model& model);
    // Returns a vector of pairs of strings, one pair per custom model instance
    std::vector<StringPair> getCustomModelNamesAndDescriptions(const Specification::Model& model);

    bool hasFP16Weights(const Specification::Model& model);
    bool hasAppleImageFeatureExtractor(const Specification::Model& model);
    bool isIOS12NeuralNetworkLayer(const Specification::NeuralNetworkLayer& layer);

    google::protobuf::RepeatedPtrField<CoreML::Specification::NeuralNetworkLayer> const *getNNSpec(const CoreML::Specification::Model& model);
}


#endif
