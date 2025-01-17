#!/bin/sh

# Pytest coremltools tests with one list of test cases to deselect and another list to save & xfail
# Usage
#     sh pytest_with_deselect_and_save.sh deselect_list_file debug_save_mlmodel_list_file [usual pytest args ...]
#
# For example, say we would like to test crop_resize op, and we want to
#     1. skip some test cases (e.g. because they have segfault)
#     2. serialize .mlpackage for some other test cases (so we can debug)
# Then we may prepare one file pytest-deselect-list.txt, specifying the to-be-deselected test cases
#     coremltools/converters/mil/mil/ops/tests/iOS14/test_image_resizing.py::TestCropResize::test_builder_to_backend_smoke\[compute_unit=ComputeUnit.CPU_ONLY-backend=BackendConfig"(backend='mlprogram', precision='fp16', opset_version=<AvailableTarget.iOS15: 6>)"-is_symbolic=True-mode=0\]
#     coremltools/converters/mil/mil/ops/tests/iOS14/test_image_resizing.py::TestCropResize::test_builder_to_backend_smoke\[compute_unit=ComputeUnit.CPU_ONLY-backend=BackendConfig"(backend='mlprogram', precision='fp16', opset_version=<AvailableTarget.iOS15: 6>)"-is_symbolic=True-mode=1\]
#     coremltools/converters/mil/mil/ops/tests/iOS14/test_image_resizing.py::TestCropResize::test_builder_to_backend_smoke\[compute_unit=ComputeUnit.CPU_ONLY-backend=BackendConfig"(backend='mlprogram', precision='fp16', opset_version=<AvailableTarget.iOS15: 6>)"-is_symbolic=True-mode=2\]
# And another file debug-save-mlmodel.txt, specifying the to-be-saved test cases
#     coremltools/converters/mil/mil/ops/tests/iOS14/test_image_resizing.py::TestCropResize::test_builder_to_backend_smoke[compute_unit=ComputeUnit.CPU_ONLY-backend=BackendConfig"(backend='mlprogram', precision='fp16', opset_version=<AvailableTarget.iOS15: 6>)"-is_symbolic=True-mode=3]
#     coremltools/converters/mil/mil/ops/tests/iOS14/test_image_resizing.py::TestCropResize::test_builder_to_backend_smoke[compute_unit=ComputeUnit.CPU_ONLY-backend=BackendConfig"(backend='mlprogram', precision='fp16', opset_version=<AvailableTarget.iOS15: 6>)"-is_symbolic=True-mode=4]
# Then invoke this script
#     sh pytest_with_deselect_and_save.sh pytest-deselect-list.txt debug-save-mlmodel.txt test_image_resizing.py::TestCropResize::test_builder_to_backend_smoke -p no:warnings
#
# PS: The test case names can be obtained by `pytest --collect-only`


deselect_list_file=$1
debug_save_mlmodel_list_file=$2
pytest_args=${@:3}

# Read deselect list file and store all to-be-deselected test cases in array `deselects`
# They will be passed to pytest as `--deselect` argument
declare -a deselects=()
if [ -f ${deselect_list_file} ]; then
    while read -r line; do
        if [[ ${line:0:1} != "#" ]]; then
            deselects+=("$line")
        fi
    done < ${deselect_list_file}
fi

# CoreMLTools pytest scripts use environmental variable DEBUG_SAVE_MLMODEL
# to specify the debug save mlmodel list file to read from
export DEBUG_SAVE_MLMODEL=${debug_save_mlmodel_list_file}

pytest_cmd="pytest "${pytest_args}
for deselect in ${deselects[@]}; do
    pytest_cmd+=" --deselect $deselect"
done
eval ${pytest_cmd}
