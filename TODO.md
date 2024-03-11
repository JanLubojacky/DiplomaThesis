# TODOs

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
