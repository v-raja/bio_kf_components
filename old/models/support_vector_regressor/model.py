from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def train(
    inputs: Input[Dataset],
    labels: Input[Dataset],
    trained_model: Output[Model],
):
    from sklearn.svm import SVR
    import numpy as np
    import joblib

    X = np.loadtxt(inputs.path)
    y = np.loadtxt(labels.path)

    model = SVR(
        C=30,
    ).fit(X, y)
    joblib.dump(model, trained_model.path)
