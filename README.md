# DiplomaThesis

## Progress since last meeting

- \[x\] improved preprocessing
  - log transform
  - variation filtering
  - read count filtering q?
  - this improved accuracy of baseline methods slightly
- able to built the interaction graphs
  - using bioGRID, miRDB and string
  - q? Is there a way to quantify the interactions?
- \[x\] spent some time trying to get GLUE to work, this was not very successful
- \[ \] found new methods which are designed to use interaction networks, this seems more promising, but it is not finished yet, need ~one more week
  - [A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification](https://arxiv.org/pdf/2302.12838.pdf)
  - [MPK-GNN](https://ieeexplore.ieee.org/document/10148642/references#references)

## Introduction & literature

- what could this analysis bring
- significance of learning biomarkers
- sequencing methods overview
- interaction data

## Datasets used

- BRCA:
  - analysis of breast cancer tissue and its classification into 5 subtypes, normal, basal-like, luminal A, luminal B and HER2-enriched
  - there are two sets of labels
    - PAM50_mRNA_nature2012
      - 522 samples
      - Basal-like: 98, HER2-enriched: 58, Luminal A: 231, Luminal B: 127, Normal-like: 8
      - normal only 8 samples -> so we can safely ignore it and use only the other 4 subtypes for the classification
    - PAM50Call_RNAseq
      - 900 samples
      - Basal: 142, Her2: 67, LumA: 434, LumB: 194, Normal: 119
      - from what I was able to [find](https://groups.google.com/g/ucsc-cancer-genomics-browser/c/p3up0ZIxAj0?pli=1) these should be preliminary calls based on illumina HiSeq 2000 RNA sequencing platform and likely more noisy than the labels reported with the paper, however other works seem to use these too
  - can be obtained from http://xena.ucsc.edu/public
  - introduced in this paper from 2012 https://www.nature.com/articles/nature11412#change-history
  - seems like a classical benchmark, many of the papers use it so it seems useful to choose this to compare performance against them
- data supplied by the IDA lab at CVUT FEL
  - #TODO is there a publication or any associated info with this data?
- other datasets which could be of use
  - ROSMAP - alzheimer disease, binary task, AD (alzheimer disease) vs NC (normal controls)
    - besides BRCA probably the second most used one
  - KIPAN - liver cancer subtype classification, also ofthen used
  - LGG - lower grade glycoma classification by four grades given by WHO
    - (from the papers its not clear to me how to split the data into classes ?)
    - also often used and obtainable from http://xena.ucsc.edu/public

## Baseline methods & results

- Data preprocessing

  - RNA expression counts, raw counts vs TPMs, RPKM or FPKM, raw counts are not a good choice for comparisions between different samples, because they do not account for differences in transcript length, total number of reads per samples, and sequencing biases ... [link](https://stats.stackexchange.com/questions/561987/should-i-use-raw-counts-tpms-or-rpkm-gene-expression-values-for-training-ml-mo)

- improve preprocessing with median absolute deviation (MAD) most-variable features.

- Note about parameter tuning

  - since we are dealing with a small dataset and the variance in predictions for each fold is fairly high this makes it hard to tune the parameters as the high variance makes it harder to estimate whether some parameters are truly performing well or whether it is just a random fluctuation

**Experiment design**

- the data were log transformed via $$y=\\log_2(x+1)$$
- normalized to zero mean and unit variance
- half of the data was used for parameter tuning and the other half for the final evaluation

**KNN**

- number of neighbours estimated via cross validation
- final evaluation was done on the test set via 50-fold random-split stratified cross validation to try and obtain some reasonable estimate of the variance,  which was very high for this dataset

**SVM with linear kernel**

- 2 parameters tuned
  - C - regularization parameter, estimated via cross validation from the interval $\[10^{-6}, 10^2\]$ (sampled logarithmically)
  - number of predictors to use estimated from the set ${50, 100, 500}$

**XGBoost**

- xgboost has many parameters, as such it is benefitial tu use a library such as [optuna](CITATION) which allows the use of more advanced optimization techniques such as TPE (Tree-structured Parzen Estimator) to explore the parameter space more efficently and greatly speed up the parameter tuning.

## Graph neural networks

- **GCN equation in two steps**
  - **Aggregate information from neighbours**
    - in the easiest form this could be an average or we could use ...
    - a weighted sum $$h\_{n(v)}=\\sum\_{u\\in N(v)}w\_{u,v}h_u$$
      - where $w\_{u,v}$ is computed as a product of roots of inverse degrees $$w\_{u,v}=\\sqrt{\\frac1{d_u}}\\sqrt{\\frac1{d_v}}$$
  - **Pass the aggregated vector through a MLP**
    - $h\_{n+1(v)}=\\sigma(W h\_{n(v)})$
    - in each layer the same weighs are reused for all of the nodes
    - and usually we also use a different weight matrix for the self loop
- **Relational GCNs**
  - relation is a triplet of $$\\text{node type}\\xrightarrow{\\text{edge type}} \\text{node type}$$
  - in RGCNs we have a different weight matrix for each of these triplets, then the update equation looks like $$h_i^{l+1}=\\sigma\\left(W_0^lh^l_i+\\sum\_{r\\in R}\\sum\_{j\\in N^r_i}\\frac1{c\_{ir}}W_r^lh_j^l\\right)$$
  - and we can (and usually we do) also introduce regularization because the number of relations can be large
  - **Block diagonal matrix**
    - we allow the matrices $W_r$ to be block diagonal, to reduce the number of parameters (and allow only positions close in the embedding to interact)
  - **Basis learning**
    - i.e. we put a cap on how. many base weight matrices $V_b$ we want per layer and compose the necessary matrices $W_r$ out of them via linear combination with learned coefficients, this is useful in case we have a heterogenous graph with many relations $$W_r^l=\\sum\_{b=1}^Ba\_{rb}^lV_b^l$$
- **Attention GNNs**
  - in attention GNNs we modify the aggregation step, instead of a simple weight given by the degrees we compute attention weights $$h\_{n(v)}=\\sum\_{u\\in N(v)} \\alpha(h_u,h_v) h_u$$
  - where $\\alpha$ is a softmax of the attention scores $$\\alpha(u,v)=\\text{softmax}(a(h_u,h_v))$$
  - where $a$ can be computed as a
    - dot product
      - $a(h_u,h_v)=h_uh_v$
    - with a learnable set of parameters vector $a$ and matrix $W$ as
      - $a(h_u,h_v)=\\text{LeakyReLU}(a^T\\cdot\[Wh_u||Wh_v\])$
      - where the $\[-||-\]$ operator means concatenation of the two transformed vectors
  - which then gives the final equation for GATs as $$h_u=\\sigma\\left(\\sum\_{v\\in N_u}\\alpha\_{uv}Wh_v\\right)$$
    - where the weight matrix $W$ is the same weight matrix as the one that was used when computing the attention scores
  - we can also have multiple attention heads, meaning in each layer we have a set of weights $a^k,W^k$ for $k\\in{1,…,n}$ for multiple attention heads
  - and we can concatenate / sum the output from each head into a new vector

## Proposed method employing interaction data

- wasn't yet able to find a paper using this (besides GLUE but that is a slightly different task)

## Results

- TBA

## Results of a baseline methods

> todo: add params for XGboost in appendix

- **IDA lab dataset - Disease**
  | Method | Accuracy | F1 macro | F1 weighted | Parameters |
  | --- | --- | --- | --- | - |
  | KNN | 0.84 ± 0.10 | 0.67 ± 0.22 | 0.83 ± 0.10 | k = 3 |
  | SVM | 0.97 ± 0.05 | 0.96 ± 0.07 | 0.97 ± 0.05 | C = 0.001, 50 features |
  | XGBoost | 0.89 ± 0.06 | 0.80 ± 0.11 | 0.88 ± 0.07 |
  | MOGONET | 0.954 ± 0.039 | 0.852 ± 0.129 | 0.944 ± 0.048 |

- **IDA lab dataset - Risk**
  | Method | Accuracy | F1 macro | F1 weighted | Parameters |
  | --- | --- | --- | --- | - |
  | KNN | 0.50 ± 0.08 | 0.48 +/- 0.08 | 0.48 +/- 0.08 | k = 3 |
  | SVM | 0.57 ± 0.20 | 0.52 ± 0.18 | 0.51 ± 0.17 | C = 5, features = 500 |
  | XGBoost | 53 ± 0.08 | 0.44 ± 0.07 | 0.49 ± 0.07 |
  | MOGONET | 0.983 ± 0.028 | 0.984 ± 0.025| 0.983 ± 0.028|

- **IDA lab dataset - Mutation**
  | Method | Accuracy | F1 macro | F1 weighted | Parameters |
  | --- | --- | --- | --- | - |
  | KNN | 0.65 ± 0.07 | 0.49 +/- 0.10 | 0.62 +/- 0.07 | best k = 2 |
  | SVM | 0.69 ± 0.10 | 0.40 ± 0.15 | 0.60 ± 0.13 | best C = 0.1 , features = 500 |
  | XGBoost | 0.67 ± 0.08 | 0.39 ± 0.16 | 0.60 ± 0.09 |
  | MOGONET | 0.946 ± 0.033 | 0.932 ± 0.045| 0.942 ± 0.038 |

- **BRCA**
  | Method | Accuracy | F1 macro | F1 weighted |
  | --- | --- | --- | --- |
  | KNN | 0.74 ± 0.02 | 0.67 ± 0.02 | 0.72 ± 0.02 |
  | SVM | 0.729 ± 0.018 | 0.64 ± 0.017 | 0.702 ± 0.017 |
  | XGBoost | 0.771 ± 0.008 | 0.754 ± 0.01 | 0.701 ± 0.17 |
  | MOGONET | 0.774 ± 0.031 | 0.719 ± 0.047 | 0.768 ± 0.038 |
  | MOGLAM | 0.824 ± 0.017 | 0.831 ± 0.016| 0.807 ± 0.024 |
  | MPK-GNN (two omics only)| 0.747 ± 0.049| 0.712 ± 0.045 | 0.747 ± 0.28 |

- **ROSMAP**
  | Method | Accuracy | F1 macro | F1 weighted |
  | --- | --- | --- | --- |
  | KNN | 0.67 ± 0.03 | 0.66 ± 0.03 | 0.66 ± 0.03 |
  | SVM | | | |
  | XGBoost | | | |
  | MOGONET | 0.806 ± 0.030 | 0.805 ± 0.031  | 0.805 ± 0.030 |

- **KIPAN**
  | Method | Accuracy | F1 macro | F1 weighted |
  | --- | --- | --- | - |
  | KNN | | | |
  | SVM | | | |
  | XGBoost | | | |
  | MOGONET | | | |

- Big comparision
  | Method | Accuracy | F1 macro | F1 weighted |
  | --- | --- | --- | --- |
  | Linear SVM | 0.729 ± 0.018 | 0.64 ± 0.017 | 0.702 ± 0.017 |
  | KNN | 0.74 ± 0.02 | 0.67 ± 0.02 | 0.72 ± 0.02 |
  | XGBoost | 0.771 ± 0.008 | 0.754 ± 0.01 | 0.701 ± 0.17 |
  | Mogonet | 0.7886 ± 0.021 | 0.7740 ± 0.029 | 0.7254 ± 0.037 |
  | MPK-GNN (2-omics only)| 0.7742 ± 0.034 | 0.7821 ± 0.031 | 0.7365 ± 0.042 |
  | MOGLAM | 0.8380 ± 0.023 | 0.8456 ± 0.022 | 0.8124 ± 0.028 |
  | Li and Nabavi | 0.864 | 0.875 | - |

## List of papers which could be of use, and their comparisions

- [LRRNS](https://link.springer.com/chapter/10.1007/978-3-319-63342-8_9)
- [EMOGI](https://www.nature.com/articles/s42256-021-00325-y)
  - [github repo](https://github.com/schulter/EMOGI)
- [GNN with multiple prior knowledge](https://dr.ntu.edu.sg/bitstream/10356/171101/2/Graph%20Neural%20Networks%20with%20Multiple%20PriorKnowledge%20for%20Multi-Omics%20Data%20Analysis.pdf)
- [Making multi-omics data accessible to researchers](https://www.nature.com/articles/s41597-019-0258-4)
- [DIABLO](https://pubmed.ncbi.nlm.nih.gov/30657866/)
- [GLUE](https://www.nature.com/articles/s41587-022-01284-4)
- [MOVE](https://github.com/RasmussenLab/MOVE)
- [SALMON](https://huangzhii.github.io/SALMON/)
- [LSTM+VAE for clustering multi-omics data](https://github.com/bilalmirza8519/LSTM-VAE)
- [DeepOmix](https://pubmed.ncbi.nlm.nih.gov/34093987/)
  - [github repo](https://github.com/CancerProfiling/DeepOmix#deepomix-a-multi-omics-scalable-and-interpretable-deep-learning-framework-and-application-in-cancer-survival-analysis)
- [GCN cancer](https://www.frontiersin.org/articles/10.3389/fphy.2020.00203/full)
  - [github repo](https://github.com/RicardoRamirez2020/GCN_Cancer)
- [DeepProg](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-00930-x)
  - [github repo](https://github.com/lanagarmire/DeepProg)
- [Deep Learning-Based Multi-Omics Integration Robustly Predicts Survival in Liver Cancer](https://pubmed.ncbi.nlm.nih.gov/28982688/)
- [Multi-omics integration method based on attention deep learning network for biomedical data classification](https://www.sciencedirect.com/science/article/pii/S0169260723000445?via%3Dihub)
- [Model evaluation](https://arxiv.org/pdf/1811.12808.pdf)

## GNN papers comparision

- [Mogonet](https://www.nature.com/articles/s41467-021-23774-w)
  - acc on BRCA
    - 0.829 ± 0.018 according to the paper
    - 0.7886 ± 0.021 according to MOGLAM
- [MoGCN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8847688/)
  - [repo](https://github.com/Lifoof/MoGCN)
  - used BRCA, but the results seem a little strange? both for the baselines and the final method, did they use a variant of the data?
  - acc on BRCA
    - 90 % according to the paper
    - 81.9 % acc according to MOGLAM
- [MODILM](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10161645/)
  - essentially MOGONET with graph attention networks + MLP instead of the GCNs
  - acc on BRCA
    - 0.845
- [MOGLAM](https://www.sciencedirect.com/science/article/pii/S0010482523007680?via%3Dihub#fig4)
  - they construct the patient similarity graph dynamically via learnable parameters, meaning that the graph structure is learned
  - they use attention to learn how much should each omic data type contribute to the classification
  - and multi-headed attention to explore correlation accross different omics data
  - acc on BRCA
    - 0.838 according to the paper
  - [github repo](https://github.com/Ouyang-Dong/MOGLAM)
- [MOALDN](https://www.sciencedirect.com/science/article/abs/pii/S0169260723000445)
  - dim. red of the patient features via a MLP
  - patient correlation via the attention mechanism
  - omics correlation via multi-omics correlation discovery network
  - acc on BRCA
    - 0.8297 ± 1.35
    - according to the paper, limited comparision against other methods
- [moBRCA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10131354/)
  - claims better performance than mogonet
- [MPK-GNN](https://ieeexplore.ieee.org/document/10148642/references#references)
  - they seem to use prior knowledge in the form of interaction networks but the paper seems a little confusing
  - also the comparisions are weird, they seem to use some form of semi-supervised learning?
- [A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification](https://arxiv.org/pdf/2302.12838.pdf)
  - integration of molecular data but no source code, good results reported in MKP-GNN on the CMSC method which should be this paper
- [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808)

## Databases

[bioGRID](https://thebiogrid.org) -> for gene interactions
[miRDB](https://mirdb.org) -> for miRNA
[string](https://string-db.org) -> for protein interactions

