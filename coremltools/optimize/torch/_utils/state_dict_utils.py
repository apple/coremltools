#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Any, Dict, Mapping, NamedTuple

import torch


class AddMetadataStateDictHook:
    """
    Create a hook that will add the given keys/values in the state dict metadata of the module it is registered on
    Args:
        extra_metadata: the extra state dict to be added to the state dict
        allow_overwrite: If True, do not raise if any of the keys are already in the state dict
          and would be overwritten by the new state
    """
    def __init__(self, extra_metadata: Mapping[str, Any], allow_overwrite: bool = False):
        self.extra_metadata = extra_metadata
        self.allow_overwrite = allow_overwrite

    def __call__(
        self,
        module: torch.nn.Module,
        destination: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        for key, value in self.extra_metadata.items():
            if key in local_metadata and not self.allow_overwrite:
                raise ValueError(
                    f"Metadata key '{key}' would be overwritten as it already exists in the local_metadata dict: {local_metadata[key]}"
                )
            local_metadata[key] = value
        return destination


class LoadStateDictPostHook:
    """
    Create a hook that acts on the module after its state_dict has been loaded.
    """

    def __call__(self, module: torch.nn.Module, incompatible_keys: NamedTuple) -> None:
        pass


def _verify_state_dict(state_dict, expected_keys):
    missing_keys = []
    unexpected_keys = []
    for key in state_dict:
        if key not in expected_keys:
            unexpected_keys.append(key)
    if len(unexpected_keys) > 0:
        raise ValueError(f"Found unexpected keys {unexpected_keys} in state_dict: {state_dict}")
    for key in expected_keys:
        if key not in state_dict:
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys {missing_keys} from state_dict: {state_dict}")
