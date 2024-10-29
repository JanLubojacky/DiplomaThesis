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
  - [ ] feature importance
- [ ] visualizer for creating output plots from the output table
- [ ] implement all the models to use the new interface
  - [ ] KNN
  - [ ] SVM
  - [ ] XGBoost
  - [ ] MLP
  - [ ] GNN models
- [ ] parsers for processing the experiment results
  - [ ] create some pretty seaborn graphs from that
  - [ ] create gene network graphs from the most important features
- [ ] differential expression analysis with DeSeq2 (mby could reuse the one that is already finished)

### Running experiments
- [ ] MDS disease
- [ ] MDS risk
- [ ] MDS mutation

# Notes
- A good point for discussion might be that when not enough samples is available it might be better to use a simpler model than a neural network
- What kinds of gene lengths should be used for normalizing? => lets use mean exon lenght
- Ensembl ids for rna-seq data are outdated, so retrieving gene lengths is not easy and would require significant amounts of manual annotations
- Could add one more task for mutations
  - SPL (splicing), EPI (epigenetic), WT (wild type), CTR (control)
- Could encode the strand as an additional feature, might be especially useful for feature based gnns where each node is a feature