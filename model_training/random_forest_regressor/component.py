from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def random_forest_regressor(
        inputs_dataset: Input[Dataset],
        labels_dataset: Input[Dataset],
        trained_model: Output[Model],
        n_estimators: int = 100,
        max_depth: int = 0,
        min_samples_split: float = 2.0,
        min_samples_leaf: float = 1.0,
):
    """
    Trains a Random Forest Regressor model.
    :param inputs_dataset: The dataset of inputs to train on.
    :param labels_dataset: The labels associated with the inputs.
    :param trained_model: The trained model.
    :param n_estimators: The number of trees in the forest.
    :param max_depth: The maximum depth of the tree. If 0, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    :param min_samples_split: The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    :return:
    """
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import joblib

    X = np.genfromtxt(inputs_dataset.path, delimiter=",")
    y = np.genfromtxt(labels_dataset.path, delimiter=",")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_split=int(min_samples_split) if min_samples_split.is_integer() else min_samples_split,
        min_samples_leaf=int(min_samples_leaf) if min_samples_leaf.is_integer() else min_samples_leaf,
    ).fit(X, y)

    joblib.dump(model, trained_model.path)


@dsl.component(
    packages_to_install=["scikit-learn", "numpy"],
)
def predict(
        inputs_dataset: Input[Dataset],
        trained_model: Input[Model],
        predicted_labels: Output[Dataset],
):
    """
    Predicts the labels for the given inputs.
    :param inputs_dataset: The dataset of inputs to predict on. This is usually the test set.
    :param trained_model: The trained model.
    :param predicted_labels: The predictions.
    """
    import joblib
    import numpy as np

    X = np.genfromtxt(inputs_dataset.path, delimiter=",")

    model = joblib.load(trained_model.path)
    y_pred = model.predict(X)

    np.savetxt(predicted_labels.path, y_pred)


if __name__ == "__main__":
    random_forest_regressor.component_spec.save("component.yaml")
    predict.component_spec.save("component_predict.yaml")
