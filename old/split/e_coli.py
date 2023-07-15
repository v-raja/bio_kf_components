from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy", "pandas"])
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
    import pandas as pd
    import numpy as np

    X = pd.read_csv(inputs.path)

    def seq_OH(seq):
        """
        one-hot encode given sequence
        returns a 4xlen(seq) array
        """
        mapping = dict(zip("acgt", range(4)))
        seq2 = [mapping[i] for i in seq]
        one_hot_seq = np.eye(4)[seq2]
        return np.reshape(one_hot_seq, (4, len(one_hot_seq)))

    # drop first four columns
    X = X.iloc[:, 4:]
    X = pd.concat(
        [X, pd.DataFrame(X["Sequence"].apply(lambda x: seq_OH(x).flatten()).to_list())],
        axis=1,
    )
    X = X.drop(columns=["Sequence"])

    y = pd.read_csv(labels.path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    np.savetxt(training_data.path, X_train)
    np.savetxt(training_labels.path, y_train)
    np.savetxt(test_data.path, X_test)
    np.savetxt(test_labels.path, y_test)
