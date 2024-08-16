# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os.path

import numpy as np
import pandas as pd
import requests


def load_boston():
    DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"
    LOCAL_FILE = "/tmp/bostonHousingData.txt"

    if not os.path.isfile(LOCAL_FILE):
        r = requests.get(DATA_URL)

        with open(LOCAL_FILE, 'w') as f:
            f.write(r.text)

    raw_df = pd.read_csv(LOCAL_FILE, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    data = np.array(data, order='C')
    target = raw_df.values[1::2, 2]

    feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                              'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

    return {"data": data, "target": target, "feature_names": feature_names}
