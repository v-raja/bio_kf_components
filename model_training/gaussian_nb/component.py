from kfp.dsl.io_types import Dataset, Model
from kfp.v2 import dsl

from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(
    packages_to_install=["scikit-learn", "numpy"],
)
def gaussian_naive_bayes(
        inputs_dataset: Input[Dataset],
        labels_dataset: Input[Dataset],
        trained_model: Output[Model],
        var_smoothing: float = 1e-9,
):
    """
    Trains a Gaussian Naive Bayes model.
    :param inputs_dataset: The dataset of inputs to train on.
    :param labels_dataset: The labels associated with the inputs.
    :param trained_model: The trained model.
    :param var_smoothing: The amount of smoothing to apply to the variance.
    """
    from sklearn.naive_bayes import GaussianNB
    import numpy as np
    import joblib

    X = np.genfromtxt(inputs_dataset.path, delimiter=",")
    y = np.genfromtxt(labels_dataset.path, delimiter=",")

    model = GaussianNB(var_smoothing=var_smoothing).fit(X, y)
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
    gaussian_naive_bayes.component_spec.save("component.yaml")
    predict.component_spec.save("component_predict.yaml")
