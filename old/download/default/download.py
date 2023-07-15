from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Output


@dsl.component(packages_to_install=["scikit-datasets", "numpy"])
def download(
    input_dataset_uri: str,
    labels_dataset_uri: str,
    inputs: Output[Dataset],
    labels: Output[Dataset],
):
    """Downloads the dataset from the uri and stores the inputs and labels in the outputs."""
    from sklearn.datasets import load_breast_cancer
    import numpy as np

    X, y = load_breast_cancer(return_X_y=True)
    np.savetxt(inputs.path, X)
    np.savetxt(labels.path, y)
