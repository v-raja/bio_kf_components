from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["scikit-learn", "numpy"])
def sklearn_train_test_split(
    dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    # stratify: List[Union[str, int]] = None,
):
    """
    Splits the dataset into training and testing datasets.
    :param dataset: The dataset to split.
    :param train_dataset: The training dataset.
    :param test_dataset: The testing dataset.
    :param test_size: See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
    :param random_state: See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
    :param shuffle: See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
    :param stratify: See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
    """
    from sklearn.model_selection import train_test_split

    import numpy as np

    dataset = np.loadtxt(dataset.path)

    train, test = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        # stratify=stratify,
    )

    np.savetxt(train_dataset.path, train)
    np.savetxt(test_dataset.path, test)


if __name__ == "__main__":
    sklearn_train_test_split.component_spec.save("component.yaml")
