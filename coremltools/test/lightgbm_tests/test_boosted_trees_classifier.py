# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np

from coremltools._deps import _HAS_LIGHTGBM, _HAS_SKLEARN, MSG_LIGHTGBM_NOT_FOUND

if _HAS_LIGHTGBM:
    import lightgbm as lgb
    from coremltools.converters import lightgbm as lgb_converter

if _HAS_SKLEARN:
    from sklearn.datasets import make_classification


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _eval_coreml_spec(spec, X):
    """Manually traverse the CoreML TreeEnsemble spec to produce probabilities.

    Used for numerical validation without needing the CoreML runtime.
    """
    ens = spec.treeEnsembleClassifier.treeEnsemble
    node_map = {(n.treeId, n.nodeId): n for n in ens.nodes}
    LEAF = 6  # LeafNode enum value

    def eval_tree(tree_id, x):
        n = node_map[(tree_id, 0)]
        while n.nodeBehavior != LEAF:
            fv = x[n.branchFeatureIndex]
            go_true = (
                n.missingValueTracksTrueChild
                if np.isnan(fv)
                else fv <= n.branchFeatureValue
            )
            n = node_map[
                (tree_id, n.trueChildNodeId if go_true else n.falseChildNodeId)
            ]
        return n.evaluationInfo[0].evaluationValue

    num_trees = max(t for t, _ in node_map) + 1
    probs = []
    for row in X:
        total = sum(eval_tree(tid, row) for tid in range(num_trees))
        probs.append(_sigmoid(total))
    return np.array(probs)


def _eval_coreml_regressor(spec, X):
    """Manually traverse a CoreML TreeEnsembleRegressor spec to produce values."""
    ens = spec.treeEnsembleRegressor.treeEnsemble
    node_map = {(n.treeId, n.nodeId): n for n in ens.nodes}
    base = list(ens.basePredictionValue)
    LEAF = 6

    def eval_tree(tree_id, x):
        n = node_map[(tree_id, 0)]
        while n.nodeBehavior != LEAF:
            fv = x[n.branchFeatureIndex]
            go_true = (
                n.missingValueTracksTrueChild
                if np.isnan(fv)
                else fv <= n.branchFeatureValue
            )
            n = node_map[
                (tree_id, n.trueChildNodeId if go_true else n.falseChildNodeId)
            ]
        return n.evaluationInfo[0].evaluationValue

    num_trees = max(t for t, _ in node_map) + 1
    out = []
    for row in X:
        total = sum(eval_tree(tid, row) for tid in range(num_trees))
        total += base[0] if base else 0.0
        out.append(total)
    return np.array(out)


def _eval_coreml_multiclass(spec, X, n_classes):
    """Manually traverse a multiclass CoreML spec and apply softmax."""
    ens = spec.treeEnsembleClassifier.treeEnsemble
    node_map = {(n.treeId, n.nodeId): n for n in ens.nodes}
    LEAF = 6

    def eval_tree(tree_id, x):
        n = node_map[(tree_id, 0)]
        while n.nodeBehavior != LEAF:
            fv = x[n.branchFeatureIndex]
            go_true = (
                n.missingValueTracksTrueChild
                if np.isnan(fv)
                else fv <= n.branchFeatureValue
            )
            n = node_map[
                (tree_id, n.trueChildNodeId if go_true else n.falseChildNodeId)
            ]
        ev = n.evaluationInfo[0]
        return ev.evaluationIndex, ev.evaluationValue

    num_trees = max(t for t, _ in node_map) + 1
    out = []
    for row in X:
        scores = np.zeros(n_classes)
        for tid in range(num_trees):
            cls, val = eval_tree(tid, row)
            scores[cls] += val
        e = np.exp(scores - scores.max())
        out.append(e / e.sum())
    return np.array(out)


@unittest.skipIf(not _HAS_LIGHTGBM, MSG_LIGHTGBM_NOT_FOUND)
@unittest.skipIf(not _HAS_SKLEARN, "scikit-learn not installed. Skipping tests.")
class LightGBMBinaryClassifierTest(unittest.TestCase):
    """Unit tests for the LightGBM → CoreML binary classifier converter."""

    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            random_state=42,
        )
        cls.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        clf = lgb.LGBMClassifier(n_estimators=50, num_leaves=8, random_state=42)
        clf.fit(X, y, feature_name=cls.feature_names)
        cls.model = clf
        cls.X = X.astype(np.float32)
        cls.y = y

    def test_spec_type(self):
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="label",
        ).get_spec()
        self.assertIsNotNone(spec)
        self.assertEqual(spec.WhichOneof("Type"), "treeEnsembleClassifier")

    def test_class_labels(self):
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="label",
            class_labels=[0, 1],
        ).get_spec()
        labels = list(spec.treeEnsembleClassifier.int64ClassLabels.vector)
        self.assertEqual(labels, [0, 1])

    def test_tree_count(self):
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="label",
        ).get_spec()
        ens = spec.treeEnsembleClassifier.treeEnsemble
        tree_ids = {n.treeId for n in ens.nodes}
        self.assertEqual(len(tree_ids), self.model.n_estimators)

    def test_post_eval_transform(self):
        """Binary classifier must use Regression_Logistic (value 2)."""
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="label",
        ).get_spec()
        # Regression_Logistic = 2 in the enum
        self.assertEqual(
            spec.treeEnsembleClassifier.postEvaluationTransform, 2
        )

    def test_numerical_accuracy(self):
        """Converted spec must match LightGBM predictions within float32 tolerance."""
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="label",
        ).get_spec()

        lgb_probs = self.model.predict_proba(self.X[:20])[:, 1]
        coreml_probs = _eval_coreml_spec(spec, self.X[:20])

        max_err = np.max(np.abs(lgb_probs - coreml_probs))
        self.assertLess(max_err, 1e-5, f"Max abs error {max_err} exceeds 1e-5")

    def test_feature_names_from_model(self):
        """When feature_names=None, names should be taken from the booster."""
        spec = lgb_converter.convert(
            self.model,
            feature_names=None,
            target="label",
        ).get_spec()
        input_names = [inp.name for inp in spec.description.input]
        self.assertEqual(sorted(input_names), sorted(self.feature_names))

    def test_metadata(self):
        import coremltools as ct
        model = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="label",
        )
        self.assertIn(ct.models._METADATA_VERSION, model.user_defined_metadata)
        src = model.user_defined_metadata[ct.models._METADATA_SOURCE]
        self.assertIn("lightgbm", src)

    def test_booster_input(self):
        """Passing a raw Booster object should also work."""
        booster = self.model.booster_
        spec = lgb_converter.convert(
            booster,
            feature_names=self.feature_names,
            target="label",
            mode="classifier",
            class_labels=[0, 1],
        ).get_spec()
        self.assertEqual(spec.WhichOneof("Type"), "treeEnsembleClassifier")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            lgb_converter.convert(
                self.model,
                feature_names=self.feature_names,
                target="label",
                mode="invalid_mode",
            )

    def test_nondefault_sigmoid(self):
        """A binary model trained with sigmoid != 1.0 must still match.

        LightGBM's binary proba is 1/(1+exp(-sigmoid*raw)); the converter folds
        the sigmoid coefficient into the leaf values so Regression_Logistic
        reproduces it exactly.
        """
        clf = lgb.LGBMClassifier(
            n_estimators=20, num_leaves=8, sigmoid=2.0, random_state=42
        )
        clf.fit(self.X, self.y, feature_name=self.feature_names)
        spec = lgb_converter.convert(
            clf, feature_names=self.feature_names, target="label"
        ).get_spec()

        lgb_probs = clf.predict_proba(self.X[:20])[:, 1]
        coreml_probs = _eval_coreml_spec(spec, self.X[:20])
        max_err = np.max(np.abs(lgb_probs - coreml_probs))
        self.assertLess(max_err, 1e-5, f"Max abs error {max_err} exceeds 1e-5")

    def test_categorical_split_raises(self):
        """Categorical ('==') splits are unsupported and must raise clearly."""
        rng = np.random.RandomState(0)
        cat = rng.randint(0, 5, size=600).astype(np.float64)
        num = rng.randn(600)
        X = np.column_stack([cat, num])
        y = ((cat == 1) | (cat == 3)).astype(int)
        clf = lgb.LGBMClassifier(n_estimators=15, num_leaves=8, random_state=0)
        clf.fit(X, y, categorical_feature=[0])
        with self.assertRaises(NotImplementedError):
            lgb_converter.convert(clf, feature_names=["c", "x"], target="label")


@unittest.skipIf(not _HAS_LIGHTGBM, MSG_LIGHTGBM_NOT_FOUND)
@unittest.skipIf(not _HAS_SKLEARN, "scikit-learn not installed. Skipping tests.")
class LightGBMRegressorTest(unittest.TestCase):
    """Unit tests for the LightGBM → CoreML regressor converter."""

    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=300, n_features=15, random_state=42)
        cls.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        reg = lgb.LGBMRegressor(n_estimators=30, num_leaves=8, random_state=42)
        reg.fit(X, y, feature_name=cls.feature_names)
        cls.model = reg
        cls.X = X.astype(np.float32)

    def test_spec_type(self):
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="value",
            mode="regressor",
        ).get_spec()
        self.assertEqual(spec.WhichOneof("Type"), "treeEnsembleRegressor")

    def test_numerical_accuracy(self):
        """Converted regressor must match LightGBM predictions.

        Guards against losing LightGBM's boost_from_average initial score, which
        is baked into the leaf values rather than carried as a base prediction.
        """
        spec = lgb_converter.convert(
            self.model,
            feature_names=self.feature_names,
            target="value",
            mode="regressor",
        ).get_spec()

        lgb_pred = self.model.predict(self.X[:20])
        coreml_pred = _eval_coreml_regressor(spec, self.X[:20])
        max_err = np.max(np.abs(lgb_pred - coreml_pred))
        self.assertLess(max_err, 1e-3, f"Max abs error {max_err} exceeds 1e-3")


@unittest.skipIf(not _HAS_LIGHTGBM, MSG_LIGHTGBM_NOT_FOUND)
@unittest.skipIf(not _HAS_SKLEARN, "scikit-learn not installed. Skipping tests.")
class LightGBMMulticlassClassifierTest(unittest.TestCase):
    """Unit tests for the LightGBM → CoreML multiclass classifier converter."""

    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=600,
            n_features=20,
            n_informative=12,
            n_classes=4,
            random_state=1,
        )
        cls.n_classes = 4
        cls.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        clf = lgb.LGBMClassifier(n_estimators=25, num_leaves=8, random_state=1)
        clf.fit(X, y, feature_name=cls.feature_names)
        cls.model = clf
        cls.X = X.astype(np.float32)

    def test_post_eval_transform(self):
        """Multiclass classifier must use Classification_SoftMax (value 1)."""
        spec = lgb_converter.convert(
            self.model, feature_names=self.feature_names, target="label"
        ).get_spec()
        # Classification_SoftMax = 1 in the enum
        self.assertEqual(spec.treeEnsembleClassifier.postEvaluationTransform, 1)

    def test_numerical_accuracy(self):
        """Converted multiclass spec must match predict_proba within tolerance."""
        spec = lgb_converter.convert(
            self.model, feature_names=self.feature_names, target="label"
        ).get_spec()

        lgb_proba = self.model.predict_proba(self.X[:20])
        coreml_proba = _eval_coreml_multiclass(spec, self.X[:20], self.n_classes)
        max_err = np.max(np.abs(lgb_proba - coreml_proba))
        self.assertLess(max_err, 1e-5, f"Max abs error {max_err} exceeds 1e-5")


if __name__ == "__main__":
    unittest.main()
