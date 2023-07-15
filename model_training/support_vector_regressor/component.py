from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def support_vector_regressor(
        inputs_dataset: Input[Dataset],
        labels_dataset: Input[Dataset],
        trained_model: Output[Model],
        kernel: str = "rbf",
        degree: float = 3,
        gamma: str = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        c: float = 1.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: int = 200,
        max_iter: int = -1,
):
    """
    Trains a support vector regressor model.
    :param inputs_dataset: The dataset of inputs to train on. This is usually the training set.
    :param labels_dataset: The labels associated with the inputs.
    :param kernel: Please see the documentation for sklearn.svm.SVR for more details.
    :param degree: Please see the documentation for sklearn.svm.SVR for more details.
    :param gamma: Please see the documentation for sklearn.svm.SVR for more details.
    :param coef0: Please see the documentation for sklearn.svm.SVR for more details.
    :param tol: Please see the documentation for sklearn.svm.SVR for more details.
    :param c: Please see the documentation for sklearn.svm.SVR for more details.
    :param epsilon: Please see the documentation for sklearn.svm.SVR for more details.
    :param shrinking: Please see the documentation for sklearn.svm.SVR for more details.
    :param cache_size: Please see the documentation for sklearn.svm.SVR for more details.
    :param max_iter: Please see the documentation for sklearn.svm.SVR for more details.
    :return:
    """
    from sklearn.svm import SVR
    import numpy as np
    import joblib

    X = np.genfromtxt(inputs_dataset.path, delimiter=",")
    y = np.genfromtxt(labels_dataset.path, delimiter=",")

    model = SVR(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        tol=tol,
        C=c,
        epsilon=epsilon,
        shrinking=shrinking,
        cache_size=cache_size,
        verbose=True,
        max_iter=max_iter,
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
    support_vector_regressor.component_spec.save("component.yaml")
    predict.component_spec.save("component_predict.yaml")
