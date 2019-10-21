# -*- coding: utf-8 -*-
# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import os

class Feature:
    """A coremltools feature under development."""
    def __init__(self, env_var, default):
        self.env_var = env_var
        self.default = default
        self.cached_value = None

    def __call__(self):
        """Determine if the feature is enabled.

        Returns
        -------
        If ``self.env_var`` is in the environment and is ``true``, ``yes``, or
        ``1`` the result is ``True``.  Otherwise, return ``self.default``.
        """
        if self.cached_value is None:
            env_val = os.environ.get(self.env_var)
            if env_val is None:
                self.cached_value = self.default
            else:
                self.cached_value = env_val in ('true', 'yes', '1')
        return self.cached_value


class Features:
    """The collection of all coremltools features under development."""
    @classmethod
    def register(cls, name, env_var, default=False):
        """Register a feature flag.

        Parameters
        ----------
        name : str
            The name of the attribute to create in this class.
        env_var : str
            The name of the environment variable to associate with this
            feature.
        default : bool
            The default value if the variable is not present in the
            environment.
        """
        setattr(cls, name, Feature(env_var, default))


# --- Feature flags may be created with something similar to this:
# Features.register('belt', 'COREMLTOOLS_BELT')
# --- This can be queried in code under development:
# if Features.belt():
#     use_belt()
# else:
#     # legacy path
#     use_suspenders()
