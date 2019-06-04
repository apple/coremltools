# Copyright (c) 2019, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ... import SPECIFICATION_VERSION

from ...proto import FeatureTypes_pb2
from .. import datatypes

import coremltools

import numpy as np
import six as _six

class KNearestNeighborsClassifierBuilder(object):
    """
    KNearestNeighborsClassifierBuilder class to construct a CoreML KNearestNeighborsClassifier specification.

    Please see the Core ML Nearest Neighbors protobuf message for more information
    on KNearestNeighborsClassifier parameters.

    Examples
    --------
    .. sourcecode:: python
        from coremltools.models.nearest_neighbors import KNearestNeighborsClassifierBuilder
        from coremltools.models.utils import save_spec

        # Create a KNearestNeighborsClassifier model that takes 4-dimensional input data and outputs a string label.
        >>> builder = KNearestNeighborsClassifierBuilder(input_name='input',
        ...                                              output_name='output',
        ...                                              number_of_dimensions=4,
        ...                                              default_class_label='default_label')

        # save the spec by the builder
        >>> save_spec(builder.spec, 'knnclassifier.mlmodel')

    See Also
    --------
    MLModel, save_spec
    """

    _VALID_INDEX_TYPES = ['linear', 'kd_tree']

    _VALID_WEIGHTING_SCHEMES = ['uniform', 'inverse_distance']

    _VALID_DISTANCE_METRICS = ['squared_euclidean']

    # Optional parameter keys for constructor
    _PARAMETER_KEY_K = 'k'
    _PARAMETER_KEY_WEIGHTING_SCHEME = 'weighting_scheme'
    _PARAMETER_KEY_INDEX_TYPE = 'index_type'
    _PARAMETER_KEY_LEAF_SIZE = 'leaf_size'
    _PARAMETER_KEY_INPUT_TYPE = 'input_type'

    # Optional parameter default values
    _PARAMETER_DEFAULT_K = 5
    _PARAMETER_DEFAULT_WEIGHTING_SCHEME = 'uniform'
    _PARAMETER_DEFAULT_INDEX_TYPE = 'linear'
    _PARAMETER_DEFAULT_LEAF_SIZE = 30
    _PARAMETER_DEFAULT_INPUT_TYPE = 'NotSpecified'

    def __init__(self, input_name, output_name, number_of_dimensions, default_class_label, **kwargs):
        """
        Create a KNearestNeighborsClassifierBuilder object.
        :param input_name: Name of the model input
        :param output_name: Name of the output
        :param number_of_dimensions: Number of dimensions of the input data
        :param default_class_label: The default class label to use for predictions. Must be either an int64 or a string.
        :param k: Number of neighbors to use for predictions. Default = 5.
        :param weighting_scheme: Weight function used in prediction. One of 'uniform' (default) or 'inverse_distance'
        :param index_type: Algorithm to compute nearest neighbors. One of 'linear' (default), or 'kd_tree'.
        :param leaf_size: Leaf size for the kd-tree. Ignored if index type is 'linear'. Default = 30.
        """
        super(KNearestNeighborsClassifierBuilder, self).__init__()

        self.spec = coremltools.proto.Model_pb2.Model()
        self.spec.specificationVersion = coremltools._MINIMUM_NEAREST_NEIGHBORS_SPEC_VERSION

        # the model is initially empty - assume it's updatable
        self.is_updatable = True

        if number_of_dimensions <= 0:
            raise ValueError('number_of_dimensions must be >= 0')
        self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.numberOfDimensions = number_of_dimensions

        input_type = kwargs.get(self._PARAMETER_KEY_INPUT_TYPE, self._PARAMETER_DEFAULT_INPUT_TYPE)
        input_feature_type = FeatureTypes_pb2.ArrayFeatureType.FLOAT32
        if input_type == datatypes.Double:
            input_feature_type = FeatureTypes_pb2.ArrayFeatureType.DOUBLE

        input_feature = self.spec.description.input.add()
        input_feature.name = input_name
        input_feature.type.multiArrayType.dataType = input_feature_type
        input_feature.type.multiArrayType.shape.extend([number_of_dimensions])

        training_features = self.spec.description.trainingInput.add()
        training_features.name = input_name
        training_features.type.multiArrayType.dataType = input_feature_type
        training_features.type.multiArrayType.shape.extend([number_of_dimensions])

        output_label = self.spec.description.output.add()
        output_label.name = output_name
        output_label_probs = self.spec.description.output.add()
        output_label_probs.name = output_name + 'Probs'
        training_features = self.spec.description.trainingInput.add()
        training_features.name = output_name

        if self._is_valid_text_type(default_class_label):
            output_label.type.stringType.MergeFromString(b'')
            training_features.type.stringType.MergeFromString(b'')
            output_label_probs.type.dictionaryType.stringKeyType.MergeFromString(b'')
            self.spec.kNearestNeighborsClassifier.stringClassLabels.MergeFromString(b'')
            self.spec.kNearestNeighborsClassifier.defaultStringLabel = default_class_label
        elif self._is_valid_number_type(default_class_label):
            output_label.type.int64Type.MergeFromString(b'')
            training_features.type.int64Type.MergeFromString(b'')
            output_label_probs.type.dictionaryType.int64KeyType.MergeFromString(b'')
            self.spec.kNearestNeighborsClassifier.int64ClassLabels.MergeFromString(b'')
            self.spec.kNearestNeighborsClassifier.defaultInt64Label = default_class_label
        else:
            raise TypeError('default_class_label type ({}) is invalid. Must be either string or int64'.format(type(default_class_label)))

        self.spec.description.predictedFeatureName = output_label.name
        self.spec.description.predictedProbabilitiesName = output_label_probs.name

        self.k = kwargs.get(self._PARAMETER_KEY_K, self._PARAMETER_DEFAULT_K)

        self.weighting_scheme = kwargs.get(self._PARAMETER_KEY_WEIGHTING_SCHEME,
                                           self._PARAMETER_DEFAULT_WEIGHTING_SCHEME)

        index_type = kwargs.get(self._PARAMETER_KEY_INDEX_TYPE, self._PARAMETER_DEFAULT_INDEX_TYPE)
        leaf_size = kwargs.get(self._PARAMETER_KEY_LEAF_SIZE, self._PARAMETER_DEFAULT_LEAF_SIZE)
        self.set_index_type(index_type, leaf_size)

        # SED is currently the only supported distance metric
        self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.squaredEuclideanDistance.MergeFromString(b'')

    @property
    def author(self):
        """
        Get the author for the KNearestNeighborsClassifier model
        :return: the author
        """
        return self.spec.description.metadata.author

    @author.setter
    def author(self, author):
        """
        Add an author for the KNearestNeighborsClassifier model
        :param author: the author
        :return: None
        """
        self.spec.description.metadata.author = author

    @property
    def license(self):
        """
            Get the license for the KNearestNeighborsClassifier model
            :return: the license
            """
        return self.spec.description.metadata.license

    @author.setter
    def license(self, license):
        """
            Add a license for the KNearestNeighborsClassifier model
            :param license: the license
            :return: None
            """
        self.spec.description.metadata.license = license

    @property
    def description(self):
        """
        Get the description for the KNearestNeighborsClassifier model
        :return: the description
        """
        return self.spec.description.metadata.shortDescription

    @description.setter
    def description(self, description):
        """
        Add a description for the model.

        :param description: the description
        :return: None
        """
        self.spec.description.metadata.shortDescription = description

    @property
    def is_updatable(self):
        """
        Check if the KNearestNeighborsClassifier is updatable
        :return: is updatable
        """
        return self.spec.isUpdatable

    @is_updatable.setter
    def is_updatable(self, is_updatable):
        """
        Set the KNearestNeighborsClassifier to be updatable
        :param is_updatable: boolean
        :return:
        """
        self.spec.isUpdatable = is_updatable

    @property
    def weighting_scheme(self):
        """
        Get the weighting scheme for the KNearestNeighborsClassifier model
        :return: the weighting scheme
        """
        return self._weighting_scheme

    @weighting_scheme.setter
    def weighting_scheme(self, weighting_scheme):
        """
        Set the weighting scheme for the KNearestNeighborsClassifier model
        :param weighting_scheme: One of [ 'uniform', 'inverse_distance' ]
        :return: None
        """
        weighting_scheme = weighting_scheme.lower()
        if weighting_scheme not in self._VALID_WEIGHTING_SCHEMES:
            raise TypeError('Invalid weighting scheme')

        if weighting_scheme == 'inverse_distance':
            self.spec.kNearestNeighborsClassifier.inverseDistanceWeighting.MergeFromString(b'')
        else:
            self.spec.kNearestNeighborsClassifier.uniformWeighting.MergeFromString(b'')

        # storing this in the object is just a convenience
        self._weighting_scheme = weighting_scheme

    @property
    def index_type(self):
        """
        Get the index type for the KNearestNeighborsClassifier model
        :return: the index type
        """
        return self._index_type

    def set_index_type(self, index_type, leaf_size=30):
        """
        Set the index type for the KNearestNeighborsClassifier model
        :param index_type: One of [ 'linear', 'kd_tree' ]
        :param leaf_size: For kd_tree indexes, the leaf size to use (default = 30)
        :return: None
        """
        index_type = index_type.lower()
        if not index_type in self._VALID_INDEX_TYPES:
            raise TypeError('Invalid index type')

        if index_type == 'kd_tree':
            if leaf_size <= 0:
                raise TypeError('leaf_size must be > 0')
            self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.singleKdTreeIndex.leafSize = leaf_size
        else:
            self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.linearIndex.MergeFromString(b'')

        # storing this in the object is just a convenience
        self._index_type = index_type

    @property
    def leaf_size(self):
        """
        Get the leaf size for the KNearestNeighborsClassifier
        :return: the leaf size
        """
        return self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.singleKdTreeIndex.leafSize

    @leaf_size.setter
    def leaf_size(self, leaf_size):
        """
        Set the leaf size for a KNearestNeighborsClassifier model. Only for kd-tree indexes.
        :param leaf_size: the leaf size
        :return:
        """
        if leaf_size <= 0:
            raise ValueError('leaf_size must be > 0')
        self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.singleKdTreeIndex.leafSize = leaf_size

    @property
    def number_of_dimensions(self):
        """
        Get the number of dimensions of the input data for the KNearestNeighborsClassifier model
        :return: number of dimensions
        """
        return self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.numberOfDimensions

    @property
    def k(self):
        """
        Get the k value for the KNearestNeighborsClassifier model
        :return: the k value
        """
        return self.spec.kNearestNeighborsClassifier.k

    @k.setter
    def k(self, k):
        """
        Set the k value for the KNearestNeighborsClassifier model
        :param k: k
        :return: None
        """
        if k <= 0:
            raise ValueError('k must be > 0')
        self.spec.kNearestNeighborsClassifier.k = k

    def add_samples(self, data_points, labels):
        """
        Add some samples to the KNearestNeighborsClassifier model
        :param data_points: List of input data points
        :param labels: List of corresponding labels
        :return: None
        """
        if len(data_points) == 0:
            raise TypeError('data_points is empty')

        if len(labels) == 0:
            raise TypeError('labels is empty')

        if len(data_points[0]) != self.number_of_dimensions:
            raise TypeError('dimensionality of data_points != expected number of dimensions')

        if len(data_points) != len(labels):
            raise TypeError('len(data_points) !=  len(labels)')

        # Validate the types of the labels before adding any points.
        self._validate_label_types(labels)

        for data_point in data_points:
            sample = self.spec.kNearestNeighborsClassifier.nearestNeighborsIndex.floatSamples.add()
            for feature in data_point:
                sample.vector.append(feature)

        if self.spec.kNearestNeighborsClassifier.HasField("int64ClassLabels"):
            for label in labels:
                self.spec.kNearestNeighborsClassifier.int64ClassLabels.vector.append(label)
        else:
            # string labels
            for label in labels:
                self.spec.kNearestNeighborsClassifier.stringClassLabels.vector.append(label)

    def _validate_label_types(self, labels):
        """
        Ensure the label types matched the expected types.
        :param spec: the spec
        :param labels: the list of labels
        :return: None, throws a TypeError if not expeted
        """
        if self.spec.kNearestNeighborsClassifier.HasField("int64ClassLabels"):
            check_is_valid = KNearestNeighborsClassifierBuilder._is_valid_number_type
        else:
            check_is_valid = KNearestNeighborsClassifierBuilder._is_valid_text_type
        for label in labels:
            if not check_is_valid(label):
                raise TypeError('Invalid type for label: {}'.format(type(label)))

    @staticmethod
    def _is_valid_text_type(obj):
        """
        Checks if the object is a valid text type.
        :param obj: the object to check
        :return: True if a valid text type, False otherwise
        """
        return isinstance(obj, _six.string_types)

    @staticmethod
    def _is_valid_number_type(obj):
        """
        Checks if the object is a valid number type.
        :param obj: the object to check
        :return: True if a valid number type, False otherwise
        """
        return isinstance(obj, (_six.integer_types, np.integer))
