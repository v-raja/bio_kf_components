# Load the model

We can load the model, which require providing an AnnData object that is structured similarly to the one used for
training (or, in most cases, the same one):

```python
pvi = scvi.model.PEAKVI.load("trained_model", adata=adata)
```

# visualizing and analyzing the latent space

We can now use the trained model to visualize, cluster, and analyze the data. We first extract the latent representation
from the model, and save it back into our AnnData object:

```py
latent = pvi.get_latent_representation()
adata.obsm["X_PeakVI"] = latent

print(latent.shape)
```

We can now use scanpy functions to cluster and visualize our latent space:

```python
# compute the k-nearest-neighbor graph that is used in both clustering and umap algorithms
sc.pp.neighbors(adata, use_rep="X_PeakVI")
# compute the umap
sc.tl.umap(adata, min_dist=0.2)
# cluster the space (we use a lower resolution to get fewer clusters than the default)
sc.tl.leiden(adata, key_added="cluster_pvi", resolution=0.2)
```

```python
sc.pl.umap(adata, color="cluster_pvi")
```

# Differential accessibility

Finally, we can use PeakVI to identify regions that are differentially accessible. There are many different ways to run
this analysis, but the simplest is comparing one cluster against all others, or comparing two clusters to each other. In
the first case we'll be looking for marker-regions, so we'll mostly want a one-sided test (the significant regions will
only be the ones preferentially accessible in our target cluster). In the second case we'll use a two-sided test to find
regions that are differentially accessible, regardless of direction.

We demonstrate both of these next, and do this in two different ways: (1) more convenient but less flexible: using an
existing factor to group the cells, and then comparing groups. (2) more flexible: using cell indices directly.

{important}
If the data includes multiple batches, we encourage setting `batch_correction=True` so the model will sample from
multiple batches when computing the differential signal. We do this below despite the data only having a single batch,
as a demonstration.

```python
# (1.1) using a known factor to compare two clusters
## two-sided is True by default, but included here for emphasis
da_res11 = pvi.differential_accessibility(
    groupby="cluster_pvi", group1="3", group2="0", two_sided=True
)

# (1.2) using a known factor to compare a cluster against all other clusters
## if we only provide group1, group2 is all other cells by default
da_res12 = pvi.differential_accessibility(
    groupby="cluster_pvi", group1="3", two_sided=False
)

# (2.1) using indices to compare two clusters
## we can use boolean masks or integer indices for the `idx1` and `idx2` arguments
da_res21 = pvi.differential_accessibility(
    idx1=adata.obs.cluster_pvi == "3",
    idx2=adata.obs.cluster_pvi == "0",
    two_sided=True,
)
# (2.2) using indices to compare a cluster against all other clusters
## if we don't provide idx2, it uses all other cells as the contrast
da_res22 = pvi.differential_accessibility(
    idx1=np.where(adata.obs.cluster_pvi == "3"),
    two_sided=False,
)

da_res22.head()
```

Note that `da_res11` and `da_res21` are equivalent, as are `da_res12` and `da_res22`. The return value is a pandas
DataFrame
with the differential results and basic properties of the comparison:

`prob_da` in our case is the probability of cells from cluster 0 being more than 0.05 (the default minimal effect) more
accessible than cells from the rest of the data.

`is_da_fdr` is a conservative classification (True/False) of whether a region is differential accessible. This is one
way
to threshold the results.

`bayes_factor` is a statistical significance score. It doesn't have a commonly acceptable threshold (e.g 0.05 for
p-values), but we demonstrate below that it's well calibrated to the effect size.

`effect_size` is the effect size, calculated as `est_prob1` - `est_prob2`.

`emp_effect` is the empirical effect size, calculated as `emp_prob1` - `emp_prob2`.

`est_prob{1,2}` are the estimated probabilities of accessibility in `group1` and `group2`.

`emp_prob{1,2}` are the empirical probabilities of detection (how many cells in group X was the region detected in).


