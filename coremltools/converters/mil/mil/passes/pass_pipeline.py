#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import Dict, List, Optional, Set, Text, Union

from tqdm import tqdm

from coremltools import _logger as logger
from coremltools.converters._profile_utils import _profile
from coremltools.converters.mil.mil import Program
from coremltools.converters.mil.mil.passes.graph_pass import PassOption
from coremltools.converters.mil.mil.passes.helper import classproperty as _classproperty
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY

_COMMON_PASSES: List[Text] = [
    "common::lower_complex_dialect_ops",
    "common::update_output_dtypes",
    "common::cast_optimization",
    "common::noop_elimination",
    # quantization pass 1: canonicalizations
    # always start quantization passes with canonicalizations
    "common::int_op_canonicalization",  # ops that support int do not need dequantize -> op -> quantize sandwich
    "common::nullify_redundant_quantization_zero_point",  # canonicalize zero point
    # quantization pass 2: remove redundancy
    # remove redundancy after canonicalization but before anything else
    "common::dequantize_quantize_pair_elimination",
    # the main quantization passes
    "common::distributive_quantized_binary_op_scale_normalization",
    # the last quantization pass: replace const dequantize with constexpr
    # after all quantization passes, since constexpr will not be further optimized
    # before const elimination, otherwise const dequantize would get bloated
    "common::dequantize_to_constexpr",
    "common::canonicalize_quantized_lut_pattern",
    "common::const_elimination",
    "common::sanitize_input_output_names",
    "common::divide_to_multiply",
    "common::select_optimization",
    "common::add_conv_transpose_output_shape",
    "common::const_elimination",
    "common::const_deduplication",  # after all consts have been settled
    "common::loop_invariant_elimination",
    "common::remove_symbolic_reshape",
    "common::noop_elimination",
    "common::fuse_matmul_weight_bias",
    "common::fuse_linear_bias",
    "common::fuse_gelu_tanh_approximation",
    "common::fuse_gelu_exact",
    "common::fuse_leaky_relu",
    "common::rank0_expand_dims_swap",
    "common::fuse_squeeze_expand_dims",
    "common::compose_conv1d",  # compose conv1d before any other conv passes
    "common::use_reflection_padding",
    "common::merge_consecutive_paddings",
    # Should come after use_reflection_padding, which will introduce new padding layers
    "common::fuse_pad_conv",  # Should come after merge_consecutive_paddings
    "common::image_input_preprocess",
    "common::replace_stack_reshape",
    # should come before detect_concat_interleave since it may add concat
    "common::reduce_transposes",
    "common::fuse_dilated_conv",
    "common::fuse_conv_scale",
    "common::fuse_conv_bias",
    "common::fuse_onehot_matmul_to_gather",
    "common::fuse_layernorm_or_instancenorm",
    # should come after reduce_transposes, to detect instance_norm
    "common::fuse_elementwise_to_batchnorm",  # should come after fuse_layernorm_or_instancenorm
    "common::fuse_reduce_mean",  # should come after fuse_layernorm_or_instancenorm
    "common::fuse_conv_batchnorm",  # should come after fuse_elementwise_to_batchnorm
    "common::fuse_conv_scale",
    # Re-run the fuse conv scale pass after the conv and batch_norm are fused
    "common::fuse_conv_bias",
    # Re-run the fuse conv bias pass after the conv and batch_norm are fused
    "common::fuse_conv_batchnorm",
    # In some cases, we need to run conv / batch_norm fusion again after the fuse_conv_scale and fuse_conv_bias passes
    "common::detect_concat_interleave",
    "common::concat_to_pixel_shuffle",
    # should come after detect_concat_interleave and after replace_stack_reshape
    "common::fuse_prelu",
    # reduce_transpose pass should run before and after this pass (the one after will be run during the cleanup passes stage)
    "common::prelu_to_lrelu",
    "common::merge_consecutive_relus",
    "common::merge_consecutive_reshapes",
    "common::merge_consecutive_transposes",
    "common::fuse_transpose_matmul",
    # "expand_high_rank_reshape_and_transpose" must come after "common::merge_consecutive_transposes"
    "common::expand_high_rank_reshape_and_transpose",
    "common::fuse_stack_split",
    "common::reduce_transposes",
    # "remove_redundant_ops" pass should be applied towards the end, once other graph passes have done their optimizations.
    # For instance, it should come after passes such as "reduce_transpose" that can introduce redundant transposes
    # in the network (while reducing the total number of transposes), and after passes such as "fuse_layernorm_or_instancenorm"
    # which detects patterns that involve redundant ops ("sub") etc.
    "common::remove_redundant_ops",
    "common::dedup_op_and_var_names",  # Must be applied before "add_fp16_cast" because "add_fp16_cast" use unique name cache.
    "common::add_fp16_cast",  # Will be removed if compute precision is not FP16.
    "common::add_int16_cast",  # Will be removed if compute precision is not FP16.
    "common::update_output_dtypes",  # Must run again after `add_fp16_cast` and `add_int16_cast`.
    "common::const_elimination",
    "common::dead_code_elimination",
    "common::cast_optimization",
    "common::dead_code_elimination",  # must follow cast_optimization
    "common::const_elimination",
    # After all fusions have settled, start inserting state ops
    "common::canonicalize_inplace_pattern",  # always start with canonicalizations
    "common::prefer_state_in_downstream",
    "common::const_elimination",
    "common::dead_code_elimination",  # always end with dce
]

_CLEANUP_PASSES: List[Text] = [
    "common::dead_code_elimination",
    "common::const_elimination",
    "common::cast_optimization",
    "common::dead_code_elimination",  # must follow cast_optimization
    "common::const_elimination",
    "common::const_deduplication",  # after all consts have been settled
    "common::dead_code_elimination",  # come before merge_affine_dequantize_with_consecutive_ops
    "common::merge_affine_dequantize_with_consecutive_ops",  # after const_deduplication and dead_code_elimination
    "common::expand_dynamic_linear",  # if weight or bias were not merged into constexpr, then expand linear to matmul + add
    "common::fuse_transpose_matmul",  # there might be left over transpose that got created in hoping to use linear, but now can be fused back with matmul
    "common::dead_code_elimination",  # fused transposes become orphans thus can be elimianted
    "common::const_deduplication",  # additional consts may be introduced during merging dequantize and expanding linear
    "common::loop_invariant_elimination",
    "common::noop_elimination",
    "common::dedup_op_and_var_names",
    "common::reduce_transposes",  # fuse_layernorm_or_instancenorm can potentially add transposes
    "common::remove_redundant_ops",
    "common::topological_reorder",
    "common::dead_code_elimination",  # always end with dce
]

_PALETTIZATION_PASSES: List[Text] = [
    "compression::palettize_weights",
]

_SPARSIFICATION_PASSES: List[Text] = [
    "compression::prune_weights",
]

_FRONTEND_TORCH_PASSES: List[Text] = [
    "common::dead_code_elimination",
    "common::loop_invariant_elimination",
    "common::dead_code_elimination",
    "torch::torch_upsample_to_core_upsample",
    "torch::torch_tensor_assign_to_core",
]

_FRONTEND_TF1_PASSES: List[Text] = [
    "common::dead_code_elimination",
    "common::loop_invariant_elimination",
    "tensorflow::backfill_make_list_elem_type",
    # DCE to reduce tf_lstm_block outputs and allow lstm_rewrite to
    # ssa lstm
    "common::dead_code_elimination",
    # tensorflow::tf_lstm_to_core_lstm must come before
    # tensorflow::expand_tf_lstm
    "tensorflow::tf_lstm_to_core_lstm",
    "tensorflow::expand_tf_lstm",
]

_FRONTEND_TF2_PASSES: List[Text] = [
    "common::dead_code_elimination",
    "common::loop_invariant_elimination",
    # tensorflow2::remove_vacuous_cond should come before
    # tensorflow::backfill_make_list_elem_type.
    "tensorflow2::remove_vacuous_cond",
    "tensorflow::backfill_make_list_elem_type",
    # DCE to reduce tf_lstm_block outputs and allow lstm_rewrite to
    # ssa lstm
    "common::dead_code_elimination",
    # tensorflow::tf_lstm_to_core_lstm must come before
    # tensorflow::expand_tf_lstm
    "tensorflow::tf_lstm_to_core_lstm",
    "tensorflow::expand_tf_lstm",
]

_BACKEND_MIL_PASSES: List[Text] = [
    "common::const_elimination",
    "mil_backend::adjust_io_to_supported_types",
    "mil_backend::insert_image_preprocessing_ops",
    "mil_backend::fuse_activation_silu",
    "mil_backend::fuse_pow2_sqrt",
    "common::const_elimination",  # rank0_expand_dims_swap might introduce some new const tensor
    "common::const_deduplication",  # after all consts have been settled
    "common::cast_optimization",
    "common::dead_code_elimination",
    "mil_backend::sanitize_name_strings",
    "common::dedup_op_and_var_names",
    "nn_backend::handle_unused_inputs",  # must come after dce.
]

_BACKEND_NN_PASSES: List[Text] = [
    "nn_backend::decompose_conv1d",  # at the beginning of nn pass
    "nn_backend::commingle_loop_vars",
    "nn_backend::handle_return_inputs_as_outputs",
    "common::const_elimination",
    "common::const_deduplication",  # after all consts have been settled
    # "remove_redundant_ops" pass should be applied towards the end, once other graph passes have done their optimizations.
    # For instance, it should come after passes such as "reduce_transpose" that can introduce redundant transposes
    # in the network (while reducing the total number of transposes), and after passes such as "fuse_layernorm_or_instancenorm"
    # which detects patterns that involve redundant ops ("sub") etc.
    "common::remove_redundant_ops",
    "common::dead_code_elimination",
    "nn_backend::handle_unused_inputs",  # must come after dce.
    "nn_backend::alert_return_type_cast",  # must be at the end.
]


class PassPipeline:
    """
    A pipeline that contains graph passes.

    Create a default pipeline (with all default graph passes that will operate on the program):

    .. sourcecode:: python

        pipeline = ct.PassPipeline.DEFAULT

    Create an empty pipeline (this will result in no graph passes being applied to the model):

    .. sourcecode:: python

        pipeline = ct.PassPipeline.EMPTY

    Add passes to pipeline:

    .. sourcecode:: python

        pipeline = ct.PassPipeline.DEFAULT
        pipeline.append_pass("common::reduce_transposes")
        pipeline.insert_pass(index=0, pass_name="common::reduce_transposes")
        # Can also specify all passes by setting the passes of the pipeline.
        pipeline.passes = ["common::reduce_transposes", "common::add_fp16_cast"]

    Remove passes:

    .. sourcecode:: python

        # Remove a pass at a specific index.
        pipeline.remove_pass(index=10)
        # Remove passes by names.
        pipeline.remove_passes({"common::add_fp16_cast", "common::reduce_transposes"})

    Inspect passes in the pipeline:

    .. sourcecode:: python

        # Get all passes.
        pass_names = pipeline.passes
        # Find indexes of a specific pass.
        pass_indexes = [
            idx
            for idx, pass_name in enumerate(pass_names)
            if pass_names[idx] == "common::reduce_transposes"
        ]

    Set options for a specific pass:

    .. sourcecode:: python

        pipeline = ct.PassPipeline.DEFAULT
        pipeline.set_options(
            pass_name="common::const_elimination",
            options={"skip_const_by_size": "100000"},
        )
    """

    # TODO: rdar://121242189 ([Infra] Have a better way to handle predefined pass pipeline)
    _PIPELINE_NAME_TO_PASSES = {
        "default": _COMMON_PASSES + _CLEANUP_PASSES,
        "cleanup": _CLEANUP_PASSES,
        "default_palettization": _PALETTIZATION_PASSES + _COMMON_PASSES + _CLEANUP_PASSES,
        "default_sparsification": _SPARSIFICATION_PASSES + _COMMON_PASSES + _CLEANUP_PASSES,
        "empty": [],
        # Frontend pipelines.
        "frontend_milinternal": [],
        "frontend_pytorch": _FRONTEND_TORCH_PASSES,
        "frontend_tensorflow": _FRONTEND_TF1_PASSES,
        "frontend_tensorflow2": _FRONTEND_TF2_PASSES,
        # Backend pipelines.
        "backend_mlprogram": _BACKEND_MIL_PASSES,
        "backend_neuralnetwork": _BACKEND_NN_PASSES,
        "backend_milinternal": [],
    }

    def __init__(self, pass_names=None, pipeline_name="default"):
        if pass_names is None:
            pass_names = _COMMON_PASSES + _CLEANUP_PASSES
        self._pass_names: List[Text] = pass_names
        self._pass_options: Dict[Text, List[PassOption]] = dict()
        self._pipeline_name = pipeline_name

    def __str__(self):
        return self._pipeline_name

    @property
    def passes(self):
        return self._pass_names

    @passes.setter
    def passes(self, passes: List[Text]):
        for pass_name in passes:
            if pass_name not in PASS_REGISTRY:
                raise ValueError(f"The pass {pass_name} is not registered.")
        self._pass_names = list(passes)

    @property
    def pipeline_name(self):
        return self._pipeline_name

    @pipeline_name.setter
    def pipeline_name(self, pipeline_name: Text):
        self._pipeline_name = pipeline_name

    def append_pass(self, pass_name: Text):
        """Append a pass at the end of the current passes in the pipeline."""
        if pass_name not in PASS_REGISTRY:
            raise ValueError(f"The pass {pass_name} is not registered.")
        self._pass_names.append(pass_name)

    def insert_pass(self, index: int, pass_name: Text) -> None:
        """Adds a pass at a specific index"""
        if pass_name not in PASS_REGISTRY:
            raise ValueError(f"The pass {pass_name} is not registered.")
        self._pass_names.insert(index, pass_name)

    def remove_pass(self, index: int) -> None:
        """Removes a pass at a specific index."""
        del self._pass_names[index]

    def remove_passes(self, passes_names: Union[Set[Text], List[Text]]) -> None:
        """Removes all passes with specific name."""
        self._pass_names = [
            pass_name for pass_name in self._pass_names if pass_name not in passes_names
        ]

    def get_options(self, pass_name: Text) -> Optional[List[PassOption]]:
        """
        Gets options of a pass that has been set by the user. Return None if the pass doesn't have
        any associated option set by the user.
        """
        return self._pass_options.get(pass_name, None)

    def get_all_options(self) -> Dict[Text, List[PassOption]]:
        """Gets all options in the pipeline."""
        return self._pass_options

    def set_options(self, pass_name: Text, options: Dict[Text, Text], override: bool = True):
        """Sets options for a specific pass."""
        if self._pass_options.get(pass_name, None):
            if not override:
                raise ValueError(f"The pass {pass_name} already has associated options.")
            else:
                logger.warning(f"The pass {pass_name} already has associated options. Override the existing options.")

        pass_options: List[PassOption] = []
        for option_name, option_val in options.items():
            pass_option = PassOption(option_name=option_name, option_val=option_val)
            pass_options.append(pass_option)
        self._pass_options[pass_name] = pass_options

    def set_options_by_another_pipeline(self, other_pipeline: PassPipeline):
        """
        Convenience method for setting options from another pipeline's options.
        For each option in other_pipeline, set it if it's also applicable to this pipeline.
        """
        for pass_name, options in other_pipeline.get_all_options().items():
            if pass_name in self.passes:
                self._pass_options[pass_name] = options

    def validate(self):
        """Validates the pipeline (including options)."""
        pass_names_set = set(self._pass_names)
        for pass_name in self._pass_options.keys():
            if pass_name not in pass_names_set:
                raise ValueError(
                    f"This pass pipeline is not valid. The pass {pass_name} has "
                    f"associated options but it's not in the passes. Passes in this "
                    f"pipeline: {self._pass_names}"
                )

    @classmethod
    def get_pipeline(cls, pipeline_name: Text) -> PassPipeline:
        """
        Gets a pipeline based on the name. Raises an error if no pipeline is found.
        Available Pipelines are defined in _PIPELINE_NAME_TO_PASSES
        """
        if pipeline_name not in cls._PIPELINE_NAME_TO_PASSES:
            raise ValueError(
                f"There is no pipeline for `{pipeline_name}`. "
                f"Available pipelines: {cls._PIPELINE_NAME_TO_PASSES.keys()}"
            )
        # We need to copy the pass names when initialize a PassPipeline object,
        # to prevent the member functions of PassPipeline from potentially modifying the original
        # data in _PIPELINE_NAME_TO_PASSES.
        passes = list(cls._PIPELINE_NAME_TO_PASSES[pipeline_name])
        return PassPipeline(passes, pipeline_name)

    @classmethod
    def list_available_pipelines(cls) -> List[str]:
        """List all available pipelines."""
        return list(cls._PIPELINE_NAME_TO_PASSES.keys())

    """
    =======================================
    Pre-defined PassPipeline configurations
    =======================================
    """
    @_classproperty
    def EMPTY(cls) -> PassPipeline:
        """Creates an empty pipeline without any pass."""
        return PassPipeline(pass_names=[])

    @_classproperty
    def DEFAULT(cls) -> PassPipeline:
        """Creates a pipeline that the converter uses by default."""
        return cls.get_pipeline("default")

    @_classproperty
    def CLEANUP(cls) -> PassPipeline:
        """Create a pipeline that contains cleanup passes."""
        return cls.get_pipeline("cleanup")

    @_classproperty
    def DEFAULT_PALETTIZATION(cls) -> PassPipeline:
        """Create a default palettization pipeline to convert a compressed source model"""
        # We use delayed import to avoid circular import
        from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig
        pipeline = cls.get_pipeline("default_palettization")

        # set default palettization
        config = OptimizationConfig(global_config=OpPalettizerConfig(mode="unique"))
        pipeline.set_options("compression::palettize_weights", {"config": config})
        return pipeline

    @_classproperty
    def DEFAULT_PRUNING(cls) -> PassPipeline:
        """Create a default sparsification pipeline to convert a compressed source model"""
        # We use delayed import to avoid circular import
        from coremltools.optimize.coreml import OpThresholdPrunerConfig, OptimizationConfig
        pipeline = cls.get_pipeline("default_sparsification")

        # set default sparsification
        config = OptimizationConfig(
            global_config=OpThresholdPrunerConfig(
                threshold=1e-12,
            )
        )
        pipeline.set_options("compression::prune_weights", {"config": config})
        return pipeline

class PassPipelineManager:
    @staticmethod
    @_profile
    def apply_pipeline(prog: Program, pass_pipeline: PassPipeline):
        """Apply a pass pipeline to a program, which modifies the program in-place."""
        if pass_pipeline is None:
            raise ValueError("The pass_pipeline cannot be None.")

        pass_pipeline.validate()
        prog.validate()

        logger.debug(f"Program before {pass_pipeline} pipeline:\n{prog}")
        for pass_name in tqdm(
            pass_pipeline.passes,
            desc=f"Running MIL {pass_pipeline} pipeline",
            unit=" passes",
        ):
            logger.debug(f'Performing pass: "{pass_name}"')
            pass_options = pass_pipeline.get_options(pass_name)
            if pass_options is not None:
                logger.debug(
                    f"The graph pass options for {pass_name} is set to {pass_options}. "
                    f"It will change the pass behavior. Make sure the option is intended."
                )
            if pass_name.startswith("experimental::"):
                logger.warning(
                    f"The graph pass {pass_name} is under experimental development, "
                    f"and the API could be changed in the future."
                )
            graph_pass = PASS_REGISTRY[pass_name]
            graph_pass.set_options(pass_options)

            try:
                graph_pass(prog)
            except Exception as e:
                logger.error(
                    f"\n\nERROR - '{pass_name}' graph pass produces the following error:\n"
                )
                raise e  # re-raise exception

            # After dead code elimination, we should check if the program misses any essential scope info
            check_essential_scope = pass_name == "common::dead_code_elimination"
            prog.validate(check_essential_scope=check_essential_scope)
        logger.debug(f"Program after {pass_pipeline} pipeline:\n{prog}")
