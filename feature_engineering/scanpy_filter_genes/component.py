from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(
    packages_to_install=["anndata", "scanpy"],
)
def scanpy_filter_genes(
        dataset: Input[Dataset],
        transformed_dataset: Output[Dataset],
        min_cells: str = "dataset.shape[0] * 0.05",
):
    """
    Filters genes from an AnnData object.
    :param dataset: An AnnData object.
    :param transformed_dataset: The AnnData object.
    :param min_cells: Minimum number of cells that a gene must be detected in to be kept. This param is evaluated as a Python expression so can reference the dataset variable.
    """
    import scanpy as sc
    import anndata

    aadata = anndata.read_h5ad(dataset.path)
    min_cells = int(eval(min_cells))
    # in-place filtering of regions
    sc.pp.filter_genes(aadata, min_cells=min_cells)

    aadata.write(transformed_dataset.path)


if __name__ == "__main__":
    scanpy_filter_genes.component_spec.save("component.yaml")
