# Global TODOs

- [ ] Baseline experiments
  - [ ] Run on several different datasets, all the models should be evaluated the same way, e.g. a 10 fold cross validation using random splits, and with the same seed to ensure equal splits
    - [ ] MDS
    - [ ] BRCA
    - [ ] ROSMAP
    - [ ] KIPAN
    - [ ] LGG
  - [ ] Compare methods with different ratios of training / testing data
  - [ ] Should choose some consistent preprocessing steps to determine what features to use in the model
- [ ] GNN experiments: Evaluate several promising models, MOGONET, MOGLAM, Li&Nabavi v2
    - [ ] MDS
    - [ ] BRCA
    - [ ] ROSMAP
    - [ ] KIPAN
    - [ ] LGG
- handle dependencies with poetry to explore how it works :)

# Local TODOs
- [x] obtain BRCA data
- [ ] preprocess the data, use simple preprocessing which selects N most variable genes across the different omic layers
  - the preprocessing should be run only on the training fold
- obtain graph data
- [ ] work on coding the bipartite gnn model

# Random notes
- for now lets only use high scoring protein interactions, > 0.7
- high scoring miRNA interactions are over 0.6, interactions over 0.8 are most likely to be real
- we have 225 mirnas without null values
- how to split classes? batch balancing?
- should filter low var mRNAs before selection
- should apply min/max normalization before selection

# Questions for MBG class
miRNAs 3p vs 5p

