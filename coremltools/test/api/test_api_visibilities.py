# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import coremltools as ct


def _get_visible_items(d):
    return [x for x in dir(d) if not x.startswith("_")]


def _check_visible_modules(actual, expected):
    assert set(actual) == set(expected), "API mis-matched. Got %s, expected %s" % (
        actual,
        expected,
    )


EXPECTED_MODULES = [
    "ClassifierConfig",
    "ComputeUnit",
    "EnumeratedShapes",
    "ImageType",
    "RangeDim",
    "SPECIFICATION_VERSION",
    "Shape",
    "TensorType",
    "colorlayout",
    "compression_utils",
    "convert",
    "converters",
    "libcoremlpython",
    "models",
    "PassPipeline",
    "proto",
    "precision",
    "target",
    "utils",
    "version",
    "test",
    "transform",
    "libmodelpackage",
    "libmilstoragepython",
    "optimize",
]


class TestApiVisibilities:
    """Test public coremltools API visibilities."""

    def test_top_level(self):
        if not ct.utils._is_macos():
             EXPECTED_MODULES.remove("libcoremlpython")
        _check_visible_modules(_get_visible_items(ct), EXPECTED_MODULES)

    def test_utils(self):
        expected = [
            "compile_model",
            "convert_double_to_float_multiarray_type",
            "evaluate_classifier",
            "evaluate_classifier_with_probabilities",
            "evaluate_regressor",
            "evaluate_transformer",
            "make_pipeline",
            "load_spec",
            "rename_feature",
            "save_spec",
        ]
        _check_visible_modules(_get_visible_items(ct.utils), expected)

    def test_models(self):
        expected = [
            "CompiledMLModel",
            "MLModel",
            "datatypes",
            "feature_vectorizer",
            "ml_program",
            "model",
            "nearest_neighbors",
            "neural_network",
            "pipeline",
            "tree_ensemble",
            "utils",
        ]
        _check_visible_modules(_get_visible_items(ct.models), expected)

    def test_models_mlmodel(self):
        expected = [
            "author",
            "get_compiled_model_path",
            "get_spec",
            "input_description",
            "license",
            "output_description",
            "predict",
            "save",
            "short_description",
            "user_defined_metadata",
            "version",
            "weights_dir",
        ]
        _check_visible_modules(_get_visible_items(ct.models.MLModel), expected)

    def test_models_neural_network(self):
        expected = [
            "AdamParams",
            "NeuralNetworkBuilder",
            "SgdParams",
            "builder",
            "flexible_shape_utils",
            "optimization_utils",
            "printer",
            "quantization_utils",
            "spec_inspection_utils",
            "update_optimizer_utils",
            "utils",
        ]
        _check_visible_modules(_get_visible_items(ct.models.neural_network), expected)

    def test_models_neural_network_utils(self):
        expected = ["NeuralNetworkBuilder", "make_image_input", "make_nn_classifier"]
        _check_visible_modules(
            _get_visible_items(ct.models.neural_network.utils), expected
        )

    def test_models_tree_ensemble(self):
        expected = [
            "TreeEnsembleBase",
            "TreeEnsembleClassifier",
            "TreeEnsembleRegressor",
            "set_classifier_interface_params",
            "set_regressor_interface_params",
        ]
        _check_visible_modules(_get_visible_items(ct.models.tree_ensemble), expected)

    def test_models_pipeline(self):
        expected = [
            "Pipeline",
            "PipelineClassifier",
            "PipelineRegressor",
            "set_classifier_interface_params",
            "set_regressor_interface_params",
            "set_training_features",
            "set_transform_interface_params",
        ]
        _check_visible_modules(_get_visible_items(ct.models.pipeline), expected)

    def test_converters(self):
        expected = [
            "ClassifierConfig",
            "ColorLayout",
            "EnumeratedShapes",
            "ImageType",
            "RangeDim",
            "Shape",
            "TensorType",
            "convert",
            "libsvm",
            "mil",
            "sklearn",
            "xgboost",
        ]
        _check_visible_modules(_get_visible_items(ct.converters), expected)

    def test_optimize(self):
        expected = [
            "coreml",
            "torch",
        ]
        _check_visible_modules(_get_visible_items(ct.optimize), expected)

    def test_optimize_coreml(self):
        expected = [
            "OpLinearQuantizerConfig",
            "OpMagnitudePrunerConfig",
            "OpPalettizerConfig",
            "OptimizationConfig",
            "OpThresholdPrunerConfig",
            "linear_quantize_weights",
            "palettize_weights",
            "prune_weights",
            "decompress_weights",
            "get_weights_metadata",
            "CoreMLWeightMetaData",
            "CoreMLOpMetaData",
        ]
        _check_visible_modules(_get_visible_items(ct.optimize.coreml), expected)

    def test_converters_libsvm(self):
        _check_visible_modules(_get_visible_items(ct.converters.libsvm), ["convert"])

    def test_converters_sklearn(self):
        _check_visible_modules(_get_visible_items(ct.converters.sklearn), ["convert"])

    def test_converters_xgboost(self):
        _check_visible_modules(_get_visible_items(ct.converters.xgboost), ["convert"])

    def test_models_neural_network_quantization_utils(self):
        expected = [
            "AdvancedQuantizedLayerSelector",
            "MatrixMultiplyLayerSelector",
            "ModelMetrics",
            "NoiseMetrics",
            "OutputMetric",
            "QuantizedLayerSelector",
            "TopKMetrics",
            "activate_int8_int8_matrix_multiplications",
            "compare_models",
            "quantize_weights",
        ]
        _check_visible_modules(
            _get_visible_items(ct.models.neural_network.quantization_utils), expected
        )

    def test_compression_utils(self):
        expected = [
            "affine_quantize_weights",
            "palettize_weights",
            "sparsify_weights",
            "decompress_weights",
        ]
        _check_visible_modules(
            _get_visible_items(ct.compression_utils), expected
        )

    def test_models_neural_network_flexible_shape_utils(self):
        expected = [
            "NeuralNetworkImageSize",
            "NeuralNetworkImageSizeRange",
            "NeuralNetworkMultiArrayShape",
            "NeuralNetworkMultiArrayShapeRange",
            "Shape",
            "ShapeRange",
            "Size",
            "add_enumerated_image_sizes",
            "add_enumerated_multiarray_shapes",
            "add_multiarray_ndshape_enumeration",
            "set_multiarray_ndshape_range",
            "update_image_size_range",
            "update_multiarray_shape_range",
        ]
        _check_visible_modules(
            _get_visible_items(ct.models.neural_network.flexible_shape_utils), expected
        )

    def test_models_neural_network_update_optimizer_utils(self):
        expected = ["AdamParams", "Batch", "RangeParam", "SgdParams"]
        _check_visible_modules(
            _get_visible_items(ct.models.neural_network.update_optimizer_utils),
            expected,
        )

    def test_models_neural_network_optimization_utils(self):
        _check_visible_modules(
            _get_visible_items(ct.models.neural_network.optimization_utils), [],
        )
