#import <CoreML/CoreML.h>
#import "CoreMLPythonArray.h"
#import "CoreMLPython.h"
#import "CoreMLPythonUtils.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#if ! __has_feature(objc_arc)
#error "ARC is off"
#endif

namespace py = pybind11;

using namespace CoreML::Python;

Model::~Model() {
    NSError *error = nil;
    NSFileManager *fileManager = [NSFileManager defaultManager];
    [fileManager removeItemAtPath:[[compiledUrl URLByDeletingLastPathComponent] path]  error:&error];
}

Model::Model(const std::string& urlStr) {
    @autoreleasepool {
        compiledUrl = Utils::stringToNSURL(urlStr);
        NSError *error = nil;
        m_model = [MLModel modelWithContentsOfURL:compiledUrl error:&error];
        Utils::handleError(error);
    }
}

py::dict Model::predict(const py::dict& input, bool useCPUOnly) {
    @autoreleasepool {
        NSError *error = nil;
        MLDictionaryFeatureProvider *inFeatures = Utils::dictToFeatures(input, &error);
        Utils::handleError(error);
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        options.usesCPUOnly = useCPUOnly;
        id<MLFeatureProvider> outFeatures = [m_model predictionFromFeatures:static_cast<MLDictionaryFeatureProvider * _Nonnull>(inFeatures)
                                                                    options:options
                                                                      error:&error];
        Utils::handleError(error);
        return Utils::featuresToDict(outFeatures);
    }
}

Model Model::fromSpec(const std::string& urlStr) {
    @autoreleasepool {
        // Compile the model
        NSError *error = nil;
        NSURL *specUrl = Utils::stringToNSURL(urlStr);
        NSURL *compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];

        // Translate into a type that pybind11 can bridge to Python
        if (error != nil) {
            std::stringstream errmsg;
            errmsg << "Error compiling model: \"";
            errmsg << error.localizedDescription.UTF8String;
            errmsg << "\".";
            throw std::runtime_error(errmsg.str());
        }
        return Model(compiledUrl.fileSystemRepresentation);
    }
}

PYBIND11_PLUGIN(libcoremlpython) {
    py::module m("libcoremlpython", "CoreML.Framework Python bindings");

    py::class_<Model>(m, "_MLModelProxy")
        .def(py::init<const std::string&>())
        .def_static("fromSpec", &Model::fromSpec)
        .def("predict", &Model::predict);

    return m.ptr();
}

#pragma clang diagnostic pop
