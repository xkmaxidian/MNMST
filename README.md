# MNMST v1.1

## MNMST: Topology of cell networks leverages identification of spatial domains from spatial transcriptomics data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10369465.svg)](https://doi.org/10.5281/zenodo.10369465)

###  Yu Wang, Zaiyi Liu, Xiaoke Ma

MNMST is a multi-layer network model to characterize and identify spatial domains in spatial transcriptomics data by integrating gene expression and spatial location information of cells. MNMST jointly decomposes multi-layer networks to learn discriminative features of cells, and identifies spatial domains by exploiting topological structure of affinity graph of cells. The proposed multi-layer network model not only outperforms state-of-the-art baselines on benchmarking datasets, but also precisely dissect cancer-related spatial domains. Furthermore, we also find that structure of spatial domains can be precisely characterized with topology of affinity graph of cells, proving the superiority of network-based models for spatial transcriptomics data. Moreover, MNMST provides an effective and efficient strategy for integrative analysis of spatial transcriptomic data, and also is applicable for processing spatial omics data of various platforms. In all, MNMST is a desirable tool for analyzing spatial transcriptomics data to facilitate the understanding of complex tissues.

![MNMST workflow](docs/MNMST.png)

## Update

**2023-12-07: The MNMST is now fully implemented in Python and supports GPU acceleration using PyTorch!**

**2023-12-13: The MNMST Python package is now released on Pypi!**

**2024-02-20: We fixed the issue where dependency were not being insatlled during the installation of MNMST.**



##### Now Supported platforms:

![Python](docs/icons-python.png)![Python](docs/icons-pytorch.png)![Python](docs/icons-matlab.png)

# Installation

#### <font color='red'>To accelerate MNMST by using GPU: If you have an NVIDIA GPU, be sure to firstly install a version of PyTorch that supports it (We recommend Pytorch >= 2.0.1). When installing MNMST without install Pytorch previous, the CPU version of torch will be installed by default for you. Here is the [installation guide of PyTorch](https://pytorch.org/get-started/locally/).</font>

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

### 2. From GitHub (Not recommend)

```
git clone https://github.com/xkmaxidian/MNMST
cd <your dir path>/MNMST/MNMST-package
python setup.py build
python setup.py install
```

Note: sometimes this way may need users install the require packages previous.

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

- Operation System (OS): Linux 5.4.0

- Python: 3.10.12

- Python packages: scanpy=1.9.3, numpy=1.24.1, pandas=2.0.3, anndata=0.9.2, scipy=1.11.2, scikit-learn=1.3.0, torch=2.0.1, matplotlib=3.7.2, psutil=5.9.5, tqdm=4.65.0, torchvision=0.15.2, leidenalg=0.10.1

Environment 2:

- Operation System (OS): [Google Colab (Linux 6.1.58)](https://colab.research.google.com/drive/19c6g02WCcj9uJlUCuyIkPXkwrnJACeEW?usp=sharing)

- Python: 3.10.12

- Python packages: scanpy=1.9.8, numpy=1.25.2, pandas=1.5.3, anndata=0.10.5, scipy=1.11.4, scikit-learn=1.2.2, torch=2.1.0, matplotlib=3.7.1, psutil=5.9.5, tqdm=4.66.2, torchvision=0.16.0, leidenalg=0.10.2

Environment 3:

- Operation System (OS): Windows 10

- Python: 3.10.9

- Python packages: scanpy=1.9.8, numpy=1.26.4, pandas=2.2.0, anndata=0.10.5, scipy=1.12.0, scikit-learn=1.4.0, torch=2.2.0, matplotlib=3.8.3, psutil=5.9.0, tqdm=4.66.2, torchvision=0.17.0, leidenalg=0.10.2

Environment 4:

- Operation System (OS): Windows 11

- Python: 3.9.18

- Python packages: scanpy=1.9.6, numpy=1.26.0, pandas=2.1.3, anndata=0.10.3, scipy=1.11.3, scikit-learn=1.3.2, torch=2.0.1, matplotlib=3.8.2, psutil=5.9.5, tqdm=4.66.1, torchvision=0.15.2, leidenalg=0.10.1

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

Last update: 02/20/2024, version 1.1.0

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

Please consider citing the following reference:

- [https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03272-0](https://doi.org/10.1186/s13059-024-03272-0)


