# Feature Engineering / Data Preprocessing Components

Feature engineering / data preprocessing components handle transformation of the train and test dataset. These
transformations could include
a wide variety of tasks such as normalization, encoding categorical features, adding missing values, etc.

# Overview of step in ML pipelines

The data preprocessing step typically happens after the dataset has been split into a train and test set. This is
because some data preprocessing transformations use all the data available to perform the step (for e.g., normalization
requires using the min and max of the dataset), and thus including the data of the test set would skew the
transformations.

Further, a dataset requires multiple preprocessing steps. Thus, you can expect multiple data preprocessing components
to be chained together and this needs to be kept in mind when authoring components (see point 1 of guidelines below).

Sometimes, on the test dataset, you need to apply a different preprocessing step that uses some information from the
preprocessing step applied on the train dataset. For example, if you normalize the train dataset, you need to use the
same min and max values to normalize the test dataset. Thus, for the test dataset, you need to apply a different
preprocessing step. In such cases, you will need one data preprocessing component for each dataset (train and test).
In such cases, in the train dataset preprocessing component, you can save a dict of values that are required for the
test dataset
preprocessing component. The component specification below includes the details of how to do this.

Lastly, before the dataset is used to train a model, the `input_label_split` component is used to specify the columns
of the datasets to be used for inputs and the columns to be used for the labels. Thus, there is no need to remove
added columns.

# Component specification

## Same component for train and test dataset

The component specification below is used when the same preprocessing step is applied to both the train and test
dataset:
Filename: `component.yaml`
Inputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| dataset | Dataset | The dataset to apply the transformation to. |
Outputs specification:
We require one component with the following outputs:
| Name | Type | Description |
| ---- | ---- | ----------- |
| transformed_dataset | Dataset | The transformed dataset. |

While the above specification is required for the component, the component can have additional inputs and outputs.
For example, a very common additional input is the columns to apply the transformation to (there is a helper function
`expand_cols` in utils that can be used to expand the column string to a list of column indices/names).

In regard to the BioML tasks webapp, additional inputs are shown in the UI for a user to give the argument a value.
However, additional outputs are ignored.

## Different components for train and test dataset

The component specification below is used when a different component is applied to the train and test dataset:

Train dataset component specification:
Filename: `component_train.yaml`
Inputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| dataset | Dataset | The train dataset to apply the transformation to. |
Outputs specification:
We require one component with the following outputs:
| Name | Type | Description |
| ---- | ---- | ----------- |
| transformed_dataset | Dataset | The transformed dataset. |
| test_dataset_input_artifact | Artifact | A Kubeflow Artifact to be passed to the component applied to the test
dataset. |

Test dataset component specification:
Filename: `component_test.yaml`
Inputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| dataset | Dataset | The test dataset to apply the transformation to. |
| test_dataset_input_artifact | Artifact | The Kubeflow Artifact passed from component applied to the train dataset. |
Outputs specification:
| Name | Type | Description |
| ---- | ---- | ----------- |
| transformed_dataset | Dataset | The transformed dataset. |

The `test_dataset_input_artifact` is an arbitrary Kubeflow Artifact. This is used to pass information from the train
component to the test component. The train component can save a dict of values that are required for the test dataset
preprocessing component. For example, if you normalize the train dataset, you need to use the same min and max values
to normalize the test dataset. Thus, in the train component, you can save a dict of the min and max values and pass
this dict as the `test_dataset_input_artifact` to the test component. In the test component, you can load the dict
and use the values to normalize the test dataset.

While the above specification is required for the component, the component can have additional inputs and outputs.
If there are additional inputs, the train component should have the same additional inputs as the test component (
equality
is checked using the argument name and type).

In regard to the BioML tasks webapp, additional inputs are shown in the UI for a user to give the argument a value.
However, additional outputs are ignored.

# Authoring guidelines

Here are some guidelines that should be taken into account when authoring a component:

* From the viewpoint of a user of the components, if two components are independently of each other (i.e. the order in
  which you apply the components doesn't lead to different outcomes), it would be ideal if the arguments specified to
  them are not dependent on the order the components are applied. Thus, avoid adding or removing columns between the
  original columns of the dataset. If you'd like to add columns, add them to the end of the dataset. If you'd like to
  remove columns, don't.
* Kubeflow parses the name of the component by converting the name of the component function to regular case. Thus,
  if you have a component function named `my_component`, the name of the component in the UI will be `My component`.
  Similarly, the description of the component and description of the arguments will be parsed from the docstring of the
  component function. Thus, it is important to have a docstring for the component function.