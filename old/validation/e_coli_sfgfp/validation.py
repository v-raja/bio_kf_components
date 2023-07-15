from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Metrics, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def validation(
    inputs: Input[Dataset],
    labels: Input[Dataset],
    trained_model: Input[Model],
    mlpipeline_metrics: Output[Metrics],
):
    import joblib
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import json

    X = np.loadtxt(inputs.path)
    y = np.loadtxt(labels.path)

    model = joblib.load(trained_model.path)
    y_pred = model.predict(X)

    mlpipeline_metrics.log_metric("mean_absolute_error", mean_absolute_error(y, y_pred))
    mlpipeline_metrics.log_metric("mean_squared_error", mean_squared_error(y, y_pred))
    mlpipeline_metrics.log_metric("r2_score", r2_score(y, y_pred))

    metrics = {
        "metrics": [
            {
                "name": "mean_absolute_error",
                "numberValue": mean_absolute_error(y, y_pred),
                "format": "RAW",
            },
            {
                "name": "mean_squared_error",
                "numberValue": mean_squared_error(y, y_pred),
                "format": "RAW",
            },
            {
                "name": "r2_score",
                "numberValue": r2_score(y, y_pred),
                "format": "RAW",
            },
        ]
    }
    with open(mlpipeline_metrics.path, "w") as f:
        json.dump(metrics, f)
