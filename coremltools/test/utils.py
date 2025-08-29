# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os.path

import numpy as np
import pandas as pd
import requests


def load_boston():
    DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz"
    LOCAL_FILE = "/tmp/boston_housing.npz"

    if not os.path.isfile(LOCAL_FILE):
        r = requests.get(DATA_URL)

        with open(LOCAL_FILE, 'wb') as f:
            f.write(r.content)

    boston = np.load(LOCAL_FILE, allow_pickle=True)
    data = boston['x']
    target = boston['y']

    feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                              'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

    return {"data": data, "target": target, "feature_names": feature_names}
