#ifndef __MODEL_ENUMS_H
#define __MODEL_ENUMS_H
enum MLModelType: int {
    MLModelType_pipelineClassifier = 200,
    MLModelType_pipelineRegressor = 201,
    MLModelType_pipeline = 202,
    MLModelType_glmRegressor = 300,
    MLModelType_supportVectorRegressor = 301,
    MLModelType_treeEnsembleRegressor = 302,
    MLModelType_neuralNetworkRegressor = 303,
    MLModelType_bayesianProbitRegressor = 304,
    MLModelType_glmClassifier = 400,
    MLModelType_supportVectorClassifier = 401,
    MLModelType_treeEnsembleClassifier = 402,
    MLModelType_neuralNetworkClassifier = 403,
    MLModelType_kNearestNeighborsClassifier = 404,
    MLModelType_neuralNetwork = 500,
    MLModelType_itemSimilarityRecommender = 501,
    MLModelType_mlProgram = 502,
    MLModelType_customModel = 555,
    MLModelType_linkedModel = 556,
    MLModelType_classConfidenceThresholding = 560,
    MLModelType_oneHotEncoder = 600,
    MLModelType_imputer = 601,
    MLModelType_featureVectorizer = 602,
    MLModelType_dictVectorizer = 603,
    MLModelType_scaler = 604,
    MLModelType_categoricalMapping = 606,
    MLModelType_normalizer = 607,
    MLModelType_arrayFeatureExtractor = 609,
    MLModelType_nonMaximumSuppression = 610,
    MLModelType_identity = 900,
    MLModelType_textClassifier = 2000,
    MLModelType_wordTagger = 2001,
    MLModelType_visionFeaturePrint = 2002,
    MLModelType_soundAnalysisPreprocessing = 2003,
    MLModelType_gazetteer = 2004,
    MLModelType_wordEmbedding = 2005,
    MLModelType_audioFeaturePrint = 2006,
    MLModelType_serializedModel = 3000,
    MLModelType_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLModelType_Name(MLModelType x) {
    switch (x) {
        case MLModelType_pipelineClassifier:
            return "MLModelType_pipelineClassifier";
        case MLModelType_pipelineRegressor:
            return "MLModelType_pipelineRegressor";
        case MLModelType_pipeline:
            return "MLModelType_pipeline";
        case MLModelType_glmRegressor:
            return "MLModelType_glmRegressor";
        case MLModelType_supportVectorRegressor:
            return "MLModelType_supportVectorRegressor";
        case MLModelType_treeEnsembleRegressor:
            return "MLModelType_treeEnsembleRegressor";
        case MLModelType_neuralNetworkRegressor:
            return "MLModelType_neuralNetworkRegressor";
        case MLModelType_bayesianProbitRegressor:
            return "MLModelType_bayesianProbitRegressor";
        case MLModelType_glmClassifier:
            return "MLModelType_glmClassifier";
        case MLModelType_supportVectorClassifier:
            return "MLModelType_supportVectorClassifier";
        case MLModelType_treeEnsembleClassifier:
            return "MLModelType_treeEnsembleClassifier";
        case MLModelType_neuralNetworkClassifier:
            return "MLModelType_neuralNetworkClassifier";
        case MLModelType_kNearestNeighborsClassifier:
            return "MLModelType_kNearestNeighborsClassifier";
        case MLModelType_neuralNetwork:
            return "MLModelType_neuralNetwork";
        case MLModelType_itemSimilarityRecommender:
            return "MLModelType_itemSimilarityRecommender";
        case MLModelType_mlProgram:
            return "MLModelType_mlProgram";
        case MLModelType_customModel:
            return "MLModelType_customModel";
        case MLModelType_linkedModel:
            return "MLModelType_linkedModel";
        case MLModelType_classConfidenceThresholding:
            return "MLModelType_classConfidenceThresholding";
        case MLModelType_oneHotEncoder:
            return "MLModelType_oneHotEncoder";
        case MLModelType_imputer:
            return "MLModelType_imputer";
        case MLModelType_featureVectorizer:
            return "MLModelType_featureVectorizer";
        case MLModelType_dictVectorizer:
            return "MLModelType_dictVectorizer";
        case MLModelType_scaler:
            return "MLModelType_scaler";
        case MLModelType_categoricalMapping:
            return "MLModelType_categoricalMapping";
        case MLModelType_normalizer:
            return "MLModelType_normalizer";
        case MLModelType_arrayFeatureExtractor:
            return "MLModelType_arrayFeatureExtractor";
        case MLModelType_nonMaximumSuppression:
            return "MLModelType_nonMaximumSuppression";
        case MLModelType_identity:
            return "MLModelType_identity";
        case MLModelType_textClassifier:
            return "MLModelType_textClassifier";
        case MLModelType_wordTagger:
            return "MLModelType_wordTagger";
        case MLModelType_visionFeaturePrint:
            return "MLModelType_visionFeaturePrint";
        case MLModelType_soundAnalysisPreprocessing:
            return "MLModelType_soundAnalysisPreprocessing";
        case MLModelType_gazetteer:
            return "MLModelType_gazetteer";
        case MLModelType_wordEmbedding:
            return "MLModelType_wordEmbedding";
        case MLModelType_audioFeaturePrint:
            return "MLModelType_audioFeaturePrint";
        case MLModelType_serializedModel:
            return "MLModelType_serializedModel";
        case MLModelType_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

#endif
