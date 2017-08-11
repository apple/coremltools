
#import "CoreMLPythonArray.h"
#import "CoreMLPython.h"
#import "CoreMLPythonUtils.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

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

static NSString* _cached_compiler_path = nil;

NSString * getCompilerPath()
{
    if(_cached_compiler_path != nil) {
        return _cached_compiler_path;
    }
    
    // For AppleInternal builds
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL isDir = NO;
    if ([fm fileExistsAtPath:@"/AppleInternal/" isDirectory:&isDir]) {
        if (isDir) {
            NSString *internalCompilerPath = @"/usr/local/bin/coremlcompiler";
            if ([fm fileExistsAtPath:internalCompilerPath]) {
                return internalCompilerPath;
            }
        }
    }
    
    // Otherwise we return the xcode toolchain path
    NSPipe *pipe = [NSPipe pipe];
    NSFileHandle *file = pipe.fileHandleForReading;
    NSTask *task = [[NSTask alloc] init];
    task.launchPath = @"/usr/bin/xcrun";
    task.arguments = @[@"-f", @"coremlcompiler"];
    task.standardError = pipe;
    task.standardOutput = pipe;
    [task launch];
    NSData *data = [file readDataToEndOfFile];
    [file closeFile];
    
    [task waitUntilExit];
    
    NSString *output = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    output = [output stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    
    if (task.terminationStatus != 0) {
        std::stringstream errmsg;
        errmsg << "Got non-zero exit code ";
        errmsg << task.terminationStatus;
        errmsg << " from xcrun. Output was: ";
        errmsg << output.UTF8String;
        throw std::runtime_error(errmsg.str());
    }
    
    _cached_compiler_path = output;
    return output;
}

Model Model::fromSpec(const std::string& urlStr) {
    @autoreleasepool {
        // compile to a temp dir, then load from there
        NSError *error;
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
        
        @try {
            // Get compiler path
            NSString *output = getCompilerPath();
            NSPipe *pipe = [NSPipe pipe];
            NSFileHandle *file = pipe.fileHandleForReading;
            NSTask *task = [[NSTask alloc] init];
            task.launchPath = output;
            task.arguments = @[@"compile", specUrl.path, tmp_dir];
            task.standardError = pipe;
            task.standardOutput = pipe;
            [task launch];
            NSData *data = [file readDataToEndOfFile];
            [file closeFile];
            
            [task waitUntilExit];
            output = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            output = [output stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];

            if (task.terminationStatus != 0) {
                std::stringstream errmsg;
                errmsg << "Got non-zero exit code ";
                errmsg << task.terminationStatus;
                errmsg << " from coremlcompiler: ";
                errmsg << output.UTF8String;
                throw std::runtime_error(errmsg.str());
            }
            
        }
        @catch (NSException *exception) {
         
            // translate into a type that pybind11 can bridge to Python
            std::stringstream errmsg;
            errmsg << "Encountered exception \"";
            errmsg << exception.name.UTF8String;
            errmsg << "\" while attempting to invoke coremlcompiler: \"";
            errmsg << exception.reason.UTF8String;
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
