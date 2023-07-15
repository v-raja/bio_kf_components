from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def train(
        inputs: Input[Dataset],
        labels: Input[Dataset],
        trained_model: Output[Model],
):
    """Trains the model on the dataset and stores the model in the output."""
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import joblib

    X = np.loadtxt(inputs.path)
    y = np.loadtxt(labels.path)

    model = RandomForestClassifier().fit(X, y)
    joblib.dump(model, trained_model.path)
