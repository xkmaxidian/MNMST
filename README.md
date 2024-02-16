# MNMST v1.1

## Multi-Layer Network Model leverages identification of spatial domains from spatial transcriptomics data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10369465.svg)](https://doi.org/10.5281/zenodo.10369465)

###  Yu Wang, Zaiyi Liu, Xiaoke Ma

MNMST is a multi-layer network model to characterize and identify spatial domains in spatial transcriptomics data by integrating gene expression and spatial location information of cells. MNMST jointly decomposes multi-layer networks to learn discriminative features of cells, and identifies spatial domains by exploiting topological structure of affinity graph of cells. The proposed multi-layer network model not only outperforms state-of-the-art baselines on benchmarking datasets, but also precisely dissect cancer-related spatial domains. Furthermore, we also find that structure of spatial domains can be precisely characterized with topology of affinity graph of cells, proving the superiority of network-based models for spatial transcriptomics data. Moreover, MNMST provides an effective and efficient strategy for integrative analysis of spatial transcriptomic data, and also is applicable for processing spatial omics data of various platforms. In all, MNMST is a desirable tool for analyzing spatial transcriptomics data to facilitate the understanding of complex tissues.

![MNMST workflow](docs/MNMST.png)

## Update

**2023-12-07: The MNMST is now fully implemented in Python and supports GPU acceleration using PyTorch!**

**2023-12-13: The MNMST Python package is now released on Pypi!**

**2024-02-16: We fixed the issue where dependency were not being insatlled during the installation of MNMST.**



##### Now Supported platforms:

![Python](docs/icons-python.png)![Python](docs/icons-pytorch.png)![Python](docs/icons-matlab.png)

# Installation

#### <font color='red'>To accelerate MNMST by using GPU: If you have an NVIDIA GPU, be sure to firstly install a version of PyTorch that supports it. When installing MNMST, the CPU version of torch will be installed by default for you. Here is the [installation guide of PyTorch](https://pytorch.org/get-started/locally/).</font>

#### 1. Start by using python virtual environment with [conda](https://anaconda.org/):

```
conda create --name mnmst python=3.9
conda activate mnmst
pip install mnmstpy
```

(Optional) To run the notebook files in tutorials, please ensure the Jupyter package is installed in your environment:

```
conda install -n mnmst ipykernel
python -m ipykernel install --user --name mnmst --display-name mnmst-jupyter
```

Note: If you encounter the error message "ImportError: Please install the skmisc package via `pip install --user scikit-misc`" while executing `sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)`, please execute the following command in your terminal: `pip install -i https://test.pypi.org/simple/ scikit-misc==0.2.0rc1`.

### 2. From GitHub

```
git clone https://github.com/xkmaxidian/MNMST
cd MNMST
```

## Tutorial

A jupyter Notebook of the tutorial for 10 $\times$ Visium is accessible from : 
<br>
https://github.com/xkmaxidian/MNMST/blob/main/tutorials/tutorials_mnmst.ipynb

The jupyter notebook of the tutorial for installing *mnmstpy* using pip and identifying spatial domains is accessible from:  

https://github.com/xkmaxidian/MNMST/blob/main/tutorials/tutorials_mnmstpy.ipynb

##### Furthermore, we explore the possibility of integrating morphological information from histology images. Detailed ideas, as well as tutorial files, are accessible from:

https://github.com/xkmaxidian/MNMST/blob/main/tutorials/tutorials_mnmst_histology.ipynb

<br>

We also provide the tutorials for other ST technologies, including [osmFISH](https://github.com/xkmaxidian/MNMST/blob/09127067b9/tutorials/tutorials_osmFISH.ipynb), [STARmap](https://github.com/xkmaxidian/MNMST/blob/09127067b9/tutorials/tutorials_STARmap.ipynb), and [Stereo-seq](https://github.com/xkmaxidian/MNMST/blob/09127067b9/tutorials/tutorials_stereo_seq.ipynb).

Please install **jupyter notebook** in order to open this notebook.



## Versions the software has been tested on:

Environment 1:

- System: Linux 5.4.0

- Python: 3.10.12

- Python packages: scanpy=1.9.3, numpy=1.24.1, pandas=2.0.3, anndata=0.9.2, scipy=1.11.2, scikit-learn=1.3.0, torch=2.0.1

Environment 2:

- System: Google Colab (Linux 6.1.58)

- Python: 3.10.12

- Python packages: scanpy=1.9.8, numpy=1.25.2, pandas=1.5.3, anndata=0.10.5, scipy=1.11.4, scikit-learn=1.2.2, torch=2.1.0

Environment 3:

- System: Windows 10

- Python: 3.10.9

- Python packages: scanpy=1.9.8, numpy=1.26.4, pandas=2.2.0, anndata=0.10.5, scipy=1.12.0, scikit-learn=1.4.0, torch=2.2.0

Environment 4:

- System: Windows 11

- Python: 3.9.18

- Python packages: scanpy=1.9.6, numpy=1.26.0, pandas=2.1.3, anndata=0.10.3, scipy=1.11.3, scikit-learn=1.3.2, torch=2.0.1



### Details (for source code): 

1. MNMST first employs SCANPY package to load and proprecess spatial transcriptomcis data:

```python
# load DLPFC 151675
section_id = '151675'
adata = sc.read_visium(path='Data/151675', count_file=section_id + '_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
# filter genes
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
hvg_filter = adata.var['highly_variable']
sc.pp.normalize_total(adata, inplace=True)
adata_all_genes = adata.copy()
adata = adata[:, hvg_filter]
# construct 1st-order cell spatial network
num_neighbours = 6
nbrs = NearestNeighbors(algorithm='ball_tree').fit(adata.obsm['spatial'])
weights_graph, distance_graph = generate_spatial_weights_fixed_nbrs(adata.obsm['spatial'], num_neighbours=num_neighbours, decay_type='reciprocal', nbr_object=nbrs, verbose=False)
# enhance data
nbrhood_contribution = 0.2
neighbour_agg_matrix = weights_graph @ adata.X
if sparse.issparse(adata.X):
    concatenated = sparse.hstack((adata.X, neighbour_agg_matrix), )
else:
    concatenated = np.concatenate((adata.X, neighbour_agg_matrix), axis=1,)

matrix = weighted_concatenate(zscore(adata.X, axis=0), zscore(neighbour_agg_matrix, axis=0), nbrhood_contribution)

if sparse.issparse(matrix):
    st_dev_pergene = matrix.toarray().std(axis=0)
else:
    st_dev_pergene = matrix.std(axis=0)

enhanced_data = matrix_to_adata(matrix, adata)

# PCA
sc.pp.pca(enhanced_data, n_comps=100)
low_dim_x = enhanced_data.obsm['X_pca']
```

2. Next, construct Cell Expression Network $W^{[e]}$ by using input expression data:

```python
from network import sparse_self_representation
from sklearn.metrics.pairwise import cosine_similarity

n_spot = low_dim_x.shape[0]
n_neighbor = 15
init_W = cosine_similarity(low_dim_x)
cos_init = np.zeros((n_spot, n_spot))
for i in range(n_spot):
    vec = init_W[i, :]
    distance = vec.argsort()[:: -1]
    for t in range(n_neighbor + 1):
        y = distance[t]
        cos_init[i, y] = init_W[i, y]
# We use the cosine similarity matrix to initialize Self-Representation Learning (SRL)
C = sparse_self_representation(low_dim_x.T, init_w=cos_init, alpha=1, beta=1)
```

3. After cell expression network constructed, we start joint NMF and affinity graph learning, where Z is the learned affinity graph:

```matlab
from MNMST import MNMST_representation
Z = MNMST_representation(C, weights_graph)
```

4. MNMST identify spatial domains by using Leiden algorithm based on the learned affinity graph:

```python
key_added = 'representation'
conns_key = 'representation'
dists_key = 'representation'

enhanced_data.uns[key_added] = {}
    
representation_dict = enhanced_data.uns[key_added]
    
representation_dict['connectivities_key'] = conns_key
representation_dict['distances_key'] = dists_key
representation_dict['var_names_use'] = enhanced_data.var_names.to_numpy()
    
representation_dict['params'] = {}
representation_dict['params']['method'] = 'umap'
enhanced_data.obsp['representation'] = Z
sc.tl.leiden(enhanced_data, neighbors_key='representation', resolution=1.4, key_added='MNMST_CPU')
display(enhanced_data.obs['MNMST_CPU'])
sc.pl.spatial(enhanced_data, color=['MNMST_CPU', 'Ground Truth'])
```

6. Additionally, MNMST can be accelerated by GPU. If you have an NVIDIA GPU, be sure to firstly install a version of [PyTorch](https://pytorch.org/) that supports it. Here is the [installation guide of PyTorch](https://pytorch.org/get-started/locally/).

```python
# First, we convert the numpy data into tensor
from MNMST_gpu import sparse_self_representation_torch, MNMST_representation_gpu
import torch

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
cos_init_tensor = torch.from_numpy(cos_init).float().to(device)
x_tensor = torch.from_numpy(low_dim_x.T).to(device)

C_gpu = sparse_self_representation_torch(x_tensor, init_w=cos_init_tensor, alpha=1., beta=1., device=device)

spatia_init_tensor = torch.from_numpy(weights_graph.A).float().to(device)
Z_gpu = MNMST_representation_gpu(C_gpu, spatia_init_tensor, device=device)
```

7. And finally, identify and visualize the spatial domains using SCANPY framework:

```
Z_gpu = Z_gpu.detach().cpu().numpy()

enhanced_data.obsp['representation'] = Z_gpu
sc.tl.leiden(enhanced_data, neighbors_key='representation', resolution=1.4, key_added='MNMST_GPU')
print(enhanced_data.obs['MNMST_GPU'])
sc.pl.spatial(enhanced_data, color=['MNMST_GPU', 'Ground Truth'])
```



## System Requirements

#### Python support packages  (Python>=3.9): 

scanpy, igraph, pandas, numpy, scipy, scanpy, anndata, sklearn, seaborn, torch, leidenalg, tqdm.

For more details of the used packages, please refer to 'requirements.txt' file.

##### The coding here is a generalization of the algorithm given in the paper. MNMST is written in Python programming language. To use, please clone this repository and follow the instructions provided in the README.md.



## File Descriptions:

Data/151675: Sample DLPFC 151675 dataset.

MNMST.py - The main function of MNMST (CPU).

MNMST_gpu.py - GPU version of the main function for MNMST.

tutorials_mnmst.ipynb - Tutorials for MNMST.

network.py - Auxiliary functions for the main MNMST function, including the sparse self representation learning.

## Compared spatial domain identification algorithms

Algorithms that are compared include: 

* [SCANPY](https://github.com/scverse/scanpy-tutorials)
* [Giotto](https://github.com/drieslab/Giotto)
* [BayesSpace](https://github.com/edward130603/BayesSpace)
* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [SEDR](https://github.com/JinmiaoChenLab/SEDR/)
* [DeepST](https://github.com/JiangBioLab/DeepST)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.

Last update: 02/16/2024, version 1.1.0

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

Our paper is under review.
