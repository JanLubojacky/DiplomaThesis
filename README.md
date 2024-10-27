## TODOs

### Preprocessing data
- [ ] initial preprocessing of the data
  - [ ] mrna
  - [ ] mirna
  - [ ] circrna
  - [ ] te_counts
  - [ ] pirna
    - doesnt contain all the samples and from previous results not that strong of a predictor, might be better to omit it
- [ ] preprocessor for splitting data into cross validation splits

### Experiments code
- [ ] base class for evaluating an experiment
- [ ] visualizer for creating output plots from the output table
- [ ] implement all the models to use the new interface
  - [ ] KNN
  - [ ] SVM
  - [ ] XGBoost
  - [ ] MLP
  - [ ] GNN models

### Running experiments
- [ ] MDS disease

# Notes
- A good point for discussion might be that when not enough samples is available it might be better to use a simpler model than a neural network
- What kinds of gene lengths should be used for normalizing? => lets use mean exon lenght
- Ensembl ids are generally stable -> good for mapping
