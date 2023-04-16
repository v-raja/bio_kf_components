# Evaluation components

Evaluation components are used to evaluate the performance of a trained model. The evaluation component takes in the
ground truth labels and the predicted labels and outputs the evaluation metrics.

# Overview of step in ML pipelines

The evaluation step happens after the model training step and the prediction step. Thus, before an evaluation component
is used, from the prediction step we have the predicted labels from the test inputs set.

# Component specification

## Evaluation component

Filename: `component.yaml`
Inputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| ground_truth_labels | Dataset | The ground-truth labels (from the test dataset). |
| predicted_labels | Dataset | The predicted labels (for the test dataset). |

Outputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| mlpipeline_metrics | Metrics | The Kubeflow metrics output. |

Note that the `mlpipeline_metrics` output is a requirement specified by Kubeflow for the correct logging of metrics.
In particular, the `mlpipeline_metrics` output should be a JSON file with the following format:

```
{
  "metrics": [
    {
      "name": "<metric-name>", # The name of the metric. Visualized as the column name in the runs table.
      "numberValue":  "0.55", # The value of the metric. Must be a numeric value.
      "format": "PERCENTAGE", # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
    },
    // Add as many metrics as needed. However, we recommend only logging one metric per component.
    {
      "name": "loss",
      "numberValue":  "0.4",
      "format": "RAW",
    }
  ]
}
```

# Authoring guidelines

Here are some guidelines that should be taken into account when authoring a component:

* Kubeflow parses the name of the component by converting the name of the component function to regular case. Thus,
  if you have a component function named `my_component`, the name of the component in the UI will be `My component`.
  Similarly, the description of the component and description of the arguments will be parsed from the docstring of the
  component function. Thus, it is important to have a docstring for the component function.