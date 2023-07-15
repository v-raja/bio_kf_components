from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Metrics, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy", "xgboost"])
def validation(
    inputs: Input[Dataset],
    labels: Input[Dataset],
    trained_model: Input[Model],
    mlpipeline_metrics: Output[Metrics],
):
    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import json

    X = np.loadtxt(inputs.path)
    y = np.loadtxt(labels.path)

    model = joblib.load(trained_model.path)
    y_pred = model.predict(X)

    mlpipeline_metrics.log_metric("accuracy", accuracy_score(y, y_pred))
    mlpipeline_metrics.log_metric("precision", precision_score(y, y_pred))
    mlpipeline_metrics.log_metric("recall", recall_score(y, y_pred))
    mlpipeline_metrics.log_metric("f1", f1_score(y, y_pred))

    metrics = {
        "metrics": [
            {
                "name": "accuracy",
                "numberValue": accuracy_score(y, y_pred),
                "format": "RAW",
            },
            {
                "name": "precision",
                "numberValue": precision_score(y, y_pred),
                "format": "RAW",
            },
            {
                "name": "recall",
                "numberValue": recall_score(y, y_pred),
                "format": "RAW",
            },
            {
                "name": "f1",
                "numberValue": f1_score(y, y_pred),
                "format": "RAW",
            },
        ]
    }
    with open(mlpipeline_metrics.path, "w") as f:
        json.dump(metrics, f)
