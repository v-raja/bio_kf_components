from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(
    packages_to_install=["anndata", "scvi", "pandas", "scikit-learn"],
)
def scvi_read_10x_atac(
        dataset: Input[Dataset],
        transformed_dataset: Output[Dataset],
):
    """
    Converts a 10x ATAC dataset to an AnnData object.
    :param dataset: The 10x ATAC dataset.
    :param transformed_dataset: The AnnData object.
    """
    import scvi

    adata = scvi.data.read_10x_atac(dataset.path)
    adata.write(transformed_dataset.path)


if __name__ == "__main__":
    scvi_read_10x_atac.component_spec.save("component.yaml")
