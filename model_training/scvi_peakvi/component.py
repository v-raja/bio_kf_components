from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset, Model
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(packages_to_install=["anndata", "scvi"])
def scvi_peakvi(
        inputs_dataset: Input[Dataset],
        trained_model: Output[Model],
        adata: Output[Dataset],
):
    """
    Trains a PEAKVI model.
    :param inputs_dataset: An AnnData object.
    :param trained_model: The trained model.
    :param adata: The AnnData object required when loading the trained model.
    """
    import anndata
    import scvi

    aadata = anndata.read_h5ad(inputs_dataset.path)
    scvi.model.PEAKVI.setup_anndata(aadata)
    pvi = scvi.model.PEAKVI(aadata)
    pvi.train()

    pvi.save(trained_model.path)
    aadata.write(adata.path)


if __name__ == "__main__":
    scvi_peakvi.component_spec.save("component.yaml")
