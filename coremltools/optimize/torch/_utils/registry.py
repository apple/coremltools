#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC as _ABC


class BaseRegistry(_ABC):
    """
    Base class for registries that register all subclasses automatically for ease-of-use.
    """

    # Maps from child class registry name to child class registry
    registry_map = dict()

    def __init_subclass__(cls, *args, **kwargs):
        # Adds mapping from child class registry name to empty child class registry
        BaseRegistry.registry_map[cls.__name__] = dict()

    @classmethod
    def instantiate(cls, subcls, *args, **kwargs):
        """
        Instantiates a subclass entry in the registry of the provided class.
        The registry is stored as a dictionary that maps from the subclass name
        to a freshly created instance of the subclass.

        Args:
            cls: The registry class, which is a subclass of BaseRegistry.
            subcls: The subclass to be registered in the registry class.
            args: The arguments to be used to create an instance of the subclass.
            kwargs: The keyword arguments to be used to create an instance of the subclass.

        """

        subcls_instance = subcls(*args, **kwargs)
        cls.register(subcls_instance)

    @classmethod
    def instantiate_key(cls, subcls_key, subcls, *args, **kwargs):
        """
        Instantiates a subclass entry in the registry of the provided class.
        The registry is stored as a dictionary that maps from the subclass key
        to a freshly created instance of the subclass.

        Args:
            cls: The registry class, which is a subclass of BaseRegistry.
            subcls_key: The subclass key to be used for the registry entry.
            subcls: The subclass to be registered in the registry class.
            args: The arguments to be used to create an instance of the subclass.
            kwargs: The keyword arguments to be used to create an instance of the subclass.
        """

        subcls_instance = subcls(*args, **kwargs)
        cls.register_key(subcls_key, subcls_instance)

    @classmethod
    def register(cls, subcls):
        """
        Registers subclass instance in registry of provided class.
        Uses the subclass name as the key for the registry entry.

        Args:
            cls: The registry class, which is a subclass of BaseRegistry.
            subcls: The subclass instance to register in the registry class.
        """

        registry = cls.get_registry()
        # Syntax is needed because cannot look up __name__ from class instance
        registry[subcls.__class__.__name__] = subcls

    @classmethod
    def register_key(cls, subcls_key, subcls):
        """
        Registers subclass instance in registry of provided class.
        Uses the subclass key as the key for the registry entry.

        Args:
            cls: The registry class, which is a subclass of BaseRegistry.
            subcls_key: The subclass key to be used for the registry entry.
            subcls: The subclass instance to register in the registry class.
        """
        registry = cls.get_registry()
        registry[subcls_key] = subcls

    @classmethod
    def get_registry(cls):
        """
        Looks up the registry corresponding to the provided registry class and
        returns it.

        Args:
            cls: The registry class, which is a subclass of BaseRegistry.
        """

        return BaseRegistry.registry_map[cls.__name__]

    @classmethod
    def get_registry_values(cls):
        """
        Looks up the registry corresponding to the provided registry class and
        returns its values. This is useful for List/Set style registries with
        keys generated automatically by this class.

        Args:
            cls: The registry class, which is a subclass of BaseRegistry.
        """

        return BaseRegistry.registry_map[cls.__name__].values()
