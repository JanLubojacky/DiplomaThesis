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
    - [x] sample based gnn - Po
    - [x] bipartite gnn - Po
- [ ] differential expression analysis with DeSeq2 (probably could reuse the one that is already finished)

### Visualizing outputs from experiments
- [ ] feature importance
- [ ] comparision of predictions given an omic channel, a couple of graphs to show how the predictions improve as we add more channels
- [ ] Evaluation of adding different channels to the model
  - We select XGBoost since it is the best out of the baseline models and the GNNs
  - For each we rank the omic channels based on their performance on each one and add them to the model in that order
  - And we look at the resulting performance

### Necessities
- [ ] hezky grafy pro nejdulezitejsi biomarkry
- [ ] enrichment analyzu pro nejdulezitejsi biomarkry
  - enrichment analysis takes a set of "important" (diff exp or here selected by the models) genes as input,
  a set of pathways with `pathway : {gene1, gene2, gene3}` as input and compares the fraction of genes in the differential set and in the background set for a pathway (fishers exact test + multiple testing correction) to find if there is a improportional fraction of genes that belong to a certain pathway in the differentially expressed genes
  - with this we can detect overrepresented pathways
  - it is also important to select a correct set of background genes (i.e. for MDS this will probably be genes that deal with blood)
  - [gene set enrichment analysis](https://www.youtube.com/watch?v=egO7Lt92gDY&t) - probably better than simple PEA
- [ ] interpretace a validace biomarkeru v literature
- [ ] v obrazcich anotovat clustery

### Feature importances
- [ ] use the best model from each task
- [ ] train it with all the omics / (or with the best configuration?)
- [ ] train it on all samples with the best parameters
- [ ] extract the feature importances from the model
- [ ] create the graph from the importances

### Target validation
- [mirnas](https://www.cuilab.cn/hmdd)
- [ctdbase](https://ctdbase.org/)
  - da se mozna najit linky nejdriv v adatabazi a az pak podle toho dohledat podpurne clanky
- [ ] GO terms
- [ ] Validace v literature
  - [ ] clanky -> pro vyznamne
  - [ ] databaze mds related genu
  - [ ] databaze mds related GO termu
    - propojeni s enrichment analyzou GO term

### For later
- [ ] prepocitat vsechno pro nove stazene datasety
  - [ ] BRCA
  - [ ] LGG
- [ ] pridat feature gnn
- [ ] increase the number of splits to obtain better estimates, especially useful for tasks such as mutation and risk where there is not very many samples
- [ ] add learning rate scheduler to the gnn trainer
- [ ] feature importances could be accumulated across the cross validation splits to get a more robust estimate
- [ ] use standard scaler instead of minmax scaler and recompute everything
- [ ] relationship between circrnas and mrnas https://circinteractome.nia.nih.gov/
