from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def data_split(
        inputs: Input[Dataset],
        labels: Input[Dataset],
        training_data: Output[Dataset],
        training_labels: Output[Dataset],
        test_data: Output[Dataset],
        test_labels: Output[Dataset],
):
    """Splits the dataset into training and testing datasets and stores them in the outputs."""
    from sklearn.model_selection import train_test_split

    import numpy as np

    X = np.loadtxt(inputs.path)
    y = np.loadtxt(labels.path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    np.savetxt(training_data.path, X_train)
    np.savetxt(training_labels.path, y_train)
    np.savetxt(test_data.path, X_test)
    np.savetxt(test_labels.path, y_test)
