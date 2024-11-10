## TODOs

### Preprocessing data
- [x] initial preprocessing of the data
  - [x] mrna
  - [x] mirna
  - [x] circrna
  - [x] te_counts
  - [x] pirna
    - doesnt contain all the samples and from previous results not that strong of a predictor, might be better to omit it
- [x] preprocessor for splitting data into cross validation splits

### Experiments code
- [x] base class for evaluating an experiment
  - [ ] feature importance
- [ ] implement all the models to use the new interface
  - [x] KNN
  - [x] SVM
  - [x] XGBoost
  - [x] MLP
  - [ ] GNN models
    - [ ] sample based gnn
    - [ ] bipartite gnn
    - [ ] feature based gnn
- [ ] differential expression analysis with DeSeq2 (probably could reuse the one that is already finished)

### Running experiments
- [ ] MDS disease
- [ ] MDS risk
- [ ] MDS mutation

### Visualizing outputs from experiments
- [ ] feature importance
- [ ] parsers for processing the experiment results
  - [ ] create some pretty seaborn graphs from that
  - [ ] create gene network graphs from the most important features (structural markers)
- [ ] visualizer for creating output plots from the output table
- [ ] comparision of predictions given an omic channel, a couple of graphs to show how the predictions improve as we add more channels
