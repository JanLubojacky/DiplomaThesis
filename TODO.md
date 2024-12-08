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
- [ ] differential expression analysis with DeSeq2 (probably could reuse the one that is already finished)

### Hyperparameter and model selection comparision
- [ ] should improve the hyperparameter search report for the gnn models, this should also probably be done on each dataset separately
- [ ] similarily the comparision of different model architectures and setups could also be improved

### Repeat experiments across datasets
- [ ] BRCA
- [ ] LGG
- do not need to do the complete evaluation with everything if there isn't time, just the comparision of the performance of the models is enough
- should also include the mRNA, circRNA, TE counts combination for MDS dataset

### Target validation
- [mirnas](https://www.cuilab.cn/hmdd)
- [ctdbase](https://ctdbase.org/)
  - da se mozna najit linky nejdriv v databazi a az pak podle toho dohledat podpurne clanky
- [ ] GO terms
- [ ] Validace v literature
  - [ ] clanky -> pro vyznamne
  - [ ] databaze mds related genu
  - [ ] databaze mds related GO termu
    - propojeni s enrichment analyzou GO term

### For later
- [ ] increase the number of splits to obtain better estimates, especially useful for tasks such as mutation and risk where there is not very many samples
- [ ] proper validation of the GNN models
- [ ] add learning rate scheduler to the gnn trainer
- [ ] use standard scaler instead of minmax scaler and recompute everything
- [ ] relationship between circrnas and mrnas https://circinteractome.nia.nih.gov/
