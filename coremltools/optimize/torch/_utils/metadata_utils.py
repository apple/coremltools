#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional

import torch as _torch
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._utils.python_utils import DictableDataClass as _DictableDataClass

STATE_DICT_METADATA_BUFFER_PREFIX = "_COREML_"
BUFFER_NAME_SEPARATOR = "/"
METADATA_VERSION_BUFFER = (
    STATE_DICT_METADATA_BUFFER_PREFIX + BUFFER_NAME_SEPARATOR + "metadata_version"
)
METADATA_VERSION = _torch.tensor(1)


class CompressionType(Enum):
    pruning = 1
    palettization = 2
    quantization = 3

    def __str__(self):
        return self.name


@_define
class CompressionMetadata(_DictableDataClass):
    """
    Class to encapsulate and register (store as buffer in state_dict) compression metadata per parameter within a module.

    Args:
        param_name (:obj:`str`): Name of parameter corresponding to which metadata is stored.
        quantization_n_bits (:obj:`int`): The dtype to use for quantizing the weights.
        quantization_scale (:py:class:`torch.Tensor`): Quantization parameters used for scaling weights.
        zero_point (:py:class:`torch.Tensor`): Quantization parameters used for translating weights in affine
            or unsigned symmetric quantization.
        lut (:py:class:`torch.Tensor`): Look up table for palettized weights.
        palettization_scale (:py:class:`torch.Tensor`): Per channel scales used to normalize weights before being palettized.
        compression_type (:obj:`list` of :py:class:`CompressionType`): List of compression types applied to the parameter
            in the order in which they were applied.
    """

    param_name: str = _field(validator=_validators.optional(_validators.instance_of(str)))
    quantization_n_bits: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    quantization_scale: _Optional[_torch.Tensor] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(_torch.Tensor)),
    )
    zero_point: _Optional[_torch.Tensor] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(_torch.Tensor)),
    )
    lut: _Optional[_torch.Tensor] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(_torch.Tensor)),
    )
    palettization_scale: _Optional[_torch.Tensor] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(_torch.Tensor)),
    )
    vector_axis: _Optional[int] = _field(
        default=None,
        validator=_validators.optional([_validators.instance_of(int)]),
    )
    compression_type: _Optional[_List[str]] = _field(
        default=None,
        converter=lambda lst: [CompressionType[item].value for item in lst] if lst else None,
        validator=_validators.optional(
            _validators.deep_iterable(
                member_validator=_validators.instance_of(int),
                iterable_validator=_validators.instance_of(list),
            )
        ),
    )

    def register(self, module: _torch.nn.Module, override_compression_type: bool = False):
        """
        Register compression metadata as buffers in module's state_dict,
        In case of joint compression, compression_type metadata is appended to module's existing compression type, if any.
        If ``override_compression_type`` flag is set, module's existing compression type metadata is overridden.
        """
        for metadata, value in self.as_dict().items():
            if metadata == "param_name" or value is None:
                continue
            buffer_name = self._get_metadata_buffer_name(metadata)

            # Handle chaining of compression types
            if metadata == "compression_type" and not override_compression_type:
                try:
                    current_value = module.get_buffer(buffer_name)
                    value = current_value.tolist() + value
                except AttributeError:
                    # Previous value doesn't exist
                    pass

            # Wrap value as a tensor to register as a buffer in module state_dict
            if not _torch.is_tensor(value):
                value = _torch.tensor(value)

            module.register_buffer(buffer_name, value)

    def _get_metadata_buffer_name(self, metadata_key: str) -> str:
        return BUFFER_NAME_SEPARATOR.join(
            [STATE_DICT_METADATA_BUFFER_PREFIX, self.param_name, metadata_key]
        )

    @classmethod
    def from_state_dict(cls, prefixed_dict) -> _Dict[str, "CompressionMetadata"]:
        """
        Initialize per parameter CompressionMetadata from state_dict
        """
        param_to_metadata_dict = dict()
        for key, value in prefixed_dict.items():
            if key.startswith(STATE_DICT_METADATA_BUFFER_PREFIX) and key != METADATA_VERSION_BUFFER:
                prefix, param_name, metadata = key.split(BUFFER_NAME_SEPARATOR)
                if param_name not in param_to_metadata_dict:
                    param_to_metadata_dict[param_name] = {"param_name": param_name}
                # For compression type, convert tensor to list of strings
                if metadata == "compression_type":
                    value = [str(CompressionType(x)) for x in value.tolist()]
                param_to_metadata_dict[param_name][metadata] = value

        result = {
            pname: cls.from_dict(metadata) for pname, metadata in param_to_metadata_dict.items()
        }
        return result


def register_metadata_version(model: _torch.nn.Module):
    """
    Register metadata version for the model
    """
    model.register_buffer(METADATA_VERSION_BUFFER, METADATA_VERSION)
