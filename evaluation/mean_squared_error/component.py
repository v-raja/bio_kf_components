from kfp.v2 import dsl


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def mean_squared_error(
        ground_truth_labels: dsl.Input[dsl.Dataset],
        predicted_labels: dsl.Input[dsl.Dataset],
        mlpipeline_metrics: dsl.Output[dsl.Metrics],
):
    """
    Measure of the average squared difference between the predicted values and the true values in a dataset. It provides an idea of the quality of a model's predictions.
    :param ground_truth_labels: The ground-truth labels (from the test dataset).
    :param predicted_labels: The predicted labels (for the test dataset).
    :param mlpipeline_metrics: The Kubeflow metrics output.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error
    import json

    y = np.loadtxt(ground_truth_labels.path)
    y_pred = np.loadtxt(predicted_labels.path)

    metrics = {
        "metrics": [
            {
                "name": "mean_squared_error",  # The name of the metric. Should be same as the id of validation metric.
                "numberValue": mean_squared_error(y, y_pred),
                "format": "RAW",
            },
        ]
    }

    with open(mlpipeline_metrics.path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    mean_squared_error.component_spec.save("component.yaml")
