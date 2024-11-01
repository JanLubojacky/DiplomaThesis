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
- finish Dockerfile for traning with GPU so that it contains
  - configured rocm
  - all dependencies
  - working jupyter-lab
- handle dependencies with poetry to explore how it works :)
- or pixi https://prefix.dev/

# Local TODOs
- [x] obtain BRCA data
- [x] preprocess the data, use simple preprocessing which selects N most variable genes across the different omic layers
  - the preprocessing should be run only on the training fold
- obtain graph data
- [x] finish the bipartite gnn model
- [ ] extend the bipartite model to multiple omics
  - [ ] use rGAT
  - [ ] parameter tuning for the graph
  - [ ] integrating predictions
    - [ ] concat and feed into linear layer
    - [ ] concat and feed into VCDN
    - [ ] have everything in one graph

- baselines
  - reimplement SVMs with recursive feature elimination

- [ ] prepare outline of diploma thesis

# Random notes
- for now lets only use high scoring protein interactions, > 0.7
- high scoring miRNA interactions are over 0.6, interactions over 0.8 are most likely to be real
- we have 225 mirnas without null values
- how to split classes? batch balancing?
- should filter low var mRNAs before selection
- should apply min/max normalization / standardization after split selection
- runnning jupyter from docker

# On improving mogonet
- graph construction
  - instead of binary A use edge weights
  - use moglam style learned graph
- use GAT instead of GCN
```sh
jupyter-lab --ip="0.0.0.0" --port=8888 --no-browser --allow-root
```
- minibatching when using a single large graphs can be done using a neighbour loader

# Questions for MBG class
miRNAs 3p vs 5p

