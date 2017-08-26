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

Model::Model(const std::string& urlStr) {
    @autoreleasepool {
        NSURL *url = Utils::stringToNSURL(urlStr);
        NSError *error = nil;
        m_model = [MLModel modelWithContentsOfURL:url error:&error];
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
        // compile to a temp dir, then load from there
        NSError *error = nil;
        NSURL *tempDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[[NSProcessInfo processInfo] globallyUniqueString]] isDirectory:YES];
        [[NSFileManager defaultManager] createDirectoryAtURL:tempDir withIntermediateDirectories:YES attributes:nil error:&error];
        Utils::handleError(error);

        NSURL *specUrl = Utils::stringToNSURL(urlStr);

        // Get a temporary directory in which to save the compiled model.
        NSString *tmp_dir = NSTemporaryDirectory();
        NSString *model_name = [[specUrl URLByDeletingPathExtension] lastPathComponent];
        NSString *basename = [tmp_dir stringByAppendingString:[NSString stringWithFormat:@"/%@.mlmodelc", model_name]];
        NSURL *compiledUrl = [NSURL fileURLWithPath:basename isDirectory:TRUE];

        NSFileManager *fileManager = [NSFileManager defaultManager];
        if(![fileManager createDirectoryAtPath:tmp_dir withIntermediateDirectories:YES attributes:nil error:&error]) {
                // An error has occurred, do something to handle it
            NSLog(@"Failed to create directory \"%@\". Error: %@", basename, error);
        }

        // Compile the model first.
        error = nil;
        compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];

        // Translate into a type that pybind11 can bridge to Python
        if (error != nil) {
            std::stringstream errmsg;
            errmsg << "Encountered exception \"";
            errmsg << error.localizedDescription.UTF8String;
            errmsg << "\" while compiling the CoreML model. \"";
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
