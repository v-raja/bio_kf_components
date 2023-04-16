# Model Training Components

Model training components handle training a model on the train dataset. These components take in the
train dataset and the labels as inputs and output the trained model.

# Overview of step in ML pipelines

The model training step typically happens after the dataset has been split into a train and test set, each set has
been through some preprocessing / feature engineering components, and the train and test datasets have each been
further split into inputs and labels. Thus, before a model training component is used, we have the following
four datasets:

- Train inputs
- Train labels
- Test inputs
- Test labels

The model training component takes in the train inputs and train labels and outputs the trained model. For each
model training component, a predict component is also specified which uses the trained model to make predictions
on the test inputs.

# Component specification

## Model training component

Filename: `component.yaml`
Inputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| inputs_dataset | Dataset | The dataset of inputs to train on. This is usually the inputs of the train dataset. |
| labels_dataset | Dataset | The labels associated with inputs_dataset. |

Outputs specification:
We require one component with the following outputs:
| Name | Type | Description |
| ---- | ---- | ----------- |
| trained_model | Model | The trained model. |

While the above specification is required for the component, the component can have additional inputs and outputs.
Common additional inputs would be the hyperparameters of the model.

In regard to the BioML tasks webapp, additional inputs are shown in the UI for a user to give the argument a value.
However, additional outputs are ignored.

## Predict component

Filename: `component_predict.yaml`

Inputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| inputs_dataset | Dataset | The dataset of inputs to predict on. This is usually the inputs of the test dataset. |
| trained_model | Model | The trained model. |
Outputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| predicted_labels | Dataset | The predictions. |

While the above specification is required for the component, the component can have additional inputs and outputs.

In regard to the BioML tasks webapp, additional inputs and outputs are ignored.

# Authoring guidelines

Here are some guidelines that should be taken into account when authoring a component:

* Kubeflow parses the name of the component by converting the name of the component function to regular case. Thus,
  if you have a component function named `my_component`, the name of the component in the UI will be `My component`.
  Similarly, the description of the component and description of the arguments will be parsed from the docstring of the
  component function. Thus, it is important to have a docstring for the component function.