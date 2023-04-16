from kfp.v2 import dsl


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def accuracy_score(
        ground_truth_labels: dsl.Input[dsl.Dataset],
        predicted_labels: dsl.Input[dsl.Dataset],
        mlpipeline_metrics: dsl.Output[dsl.Metrics],
):
    """
    Accuracy is the proportion of correct predictions among the total number of cases processed. It can be
    computed with: Accuracy = (TP + TN) / (TP + TN + FP + FN) Where: TP: True positive TN: True negative FP: False positive
    FN: False negative.
    :param ground_truth_labels: The ground-truth labels (from the test dataset).
    :param predicted_labels: The predicted labels (for the test dataset).
    :param mlpipeline_metrics: The Kubeflow metrics output.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score
    import json

    y = np.loadtxt(ground_truth_labels.path)
    y_pred = np.loadtxt(predicted_labels.path)

    metrics = {
        "metrics": [
            {
                "name": "accuracy",  # The name of the metric. Should be same as the id of validation metric.
                "numberValue": accuracy_score(y, y_pred),
                "format": "RAW",
            },
        ]
    }

    with open(mlpipeline_metrics.path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    accuracy_score.component_spec.save("component.yaml")
