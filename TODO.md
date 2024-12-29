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
- [ ] implement all the models to use the new interface
  - [x] KNN
  - [x] SVM
  - [x] XGBoost
  - [x] MLP
  - [ ] GNN models
    - [x] sample based gnn
    - [x] bipartite gnn
    - [ ] feature gnn
        for each gene create a node with `[mrna, meth, cnv]`
        create additional nodes for mirna with `[mirna, 0, 0]`

        additionally do what is done by Bingjun Li and Sheida Nabavi in 2023
        add a parallel network (concat of all features probably easiest?)
        possibly integrate with a cnn => should be better
        we could also aggregate the mirnas in the same way methylations, (this would probably have to be dynamic to be able to get feature importance for mirnas)
        are aggregated and add edges for mirnas to the network

### Hyperparameter and model selection comparision
- [ ] could improve the hyperparameter search report for the gnn models, this should also probably be done on each dataset separately
  - [ ] e.g. should show the performance of MOGONET, BipartiteGNN and FeatureGNN changes wrt to the hyperparameters on each dataset
- [ ] similarily the comparision of different model architectures and setups could also be improved

### Repeat experiments across datasets
- [ ] BRCA
- [ ] LGG
- do not need to do the complete evaluation with everything if there isn't time, just the comparision of the performance of the models is enough
- should also include the mRNA, circRNA, TE counts combination for MDS dataset

### Feature importance
- [ ] Describe feature importance in more detail (the cross validation part isnt mentioned anywhere yet)

### Finishline
- [ ] Redo feature importances for mogonet with agrregations for disease, risk and mutation
- [ ] Evaluations for additoinal datasets
  - [ ] BRCA
  - [ ] LGG
- [ ] Add feature graph model

### Target validation
- [mirnas](https://www.cuilab.cn/hmdd)
- [ctdbase](https://ctdbase.org/)

### For later
- [ ] increase the number of splits to obtain better estimates, especially useful for tasks such as mutation and risk where there is not very many samples
- [ ] proper validation of the GNN models
- [ ] add learning rate scheduler to the gnn trainer
- [ ] use standard scaler instead of minmax scaler and recompute everything
- [ ] relationship between circrnas and mrnas https://circinteractome.nia.nih.gov/
- [ ] transcription factors should be included in the network creation
- [ ] fix the feature preprocessing, COV selection should be after log transform, since log transform is used to stabilize variance and normalize the data and COV relies on the data following normal distribution, then we do not even need to use COV and can use simple variance filtering
- [ ] high variance in feature importance for mogonet, accumulation of feature importance across multiple training runs could be a solution
- [ ] vytisteni diplomky

### Thesis chapters adjustments
- [ ] 1. Introduction
- [ ] 2. Biological background
- [ ] 3. Graph Neural Networks
- [ ] 4. Classification from omic data
- [ ] 5. State of the art GNN approaches
- [ ] 6. Methods
- [ ] 7. Data
- [ ] 8. Results
- [ ] 9. Conclusions

### Thesis notes
- cite uhkt papers with connections to MDS data
  - https://www.mdpi.com/2073-4409/9/4/794
  - https://febs.onlinelibrary.wiley.com/doi/10.1002/1878-0261.13486
- [ ] make sure that all interactions are cited
- [ ] rewrite the data preprocessing section
- [ ] MOGONET enriched pathways and gene ontologies for MDS disease
