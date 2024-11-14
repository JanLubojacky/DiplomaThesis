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
    - [ ] bipartite gnn - Po
- [ ] differential expression analysis with DeSeq2 (probably could reuse the one that is already finished)

### Running experiments
- [ ] MDS disease - Ut
- [ ] MDS risk - Ut
- [ ] MDS mutation - Ut

### Visualizing outputs from experiments
- [ ] feature importance
- [ ] parsers for processing the experiment results
  - [ ] create some pretty seaborn graphs from that
  - [ ] create gene network graphs from the most important features (structural markers)
- [ ] visualizer for creating output plots from the output table
- [ ] comparision of predictions given an omic channel, a couple of graphs to show how the predictions improve as we add more channels

### Ctvrtek necessities - St + Ct dopo
- [ ] hezky grafy pro nejdulezitejsi biomarkry
- [ ] enrichment analyzu pro nejdulezitejsi biomarkry
  - enrichment analysis takes a set of "important" (diff exp or here selected by the models) genes as input,
  a set of pathways with `pathway : {gene1, gene2, gene3}` as input and compares the fraction of genes in the differential set and in the background set for a pathway (fishers exact test + multiple testing correction) to find if there is a improportional fraction of genes that belong to a certain pathway in the differentially expressed genes
  - with this we can detect overrepresented pathways
  - it is also important to select a correct set of background genes (i.e. for MDS this will probably be genes that deal with blood)
  - [gene set enrichment analysis](https://www.youtube.com/watch?v=egO7Lt92gDY&t) - probably better than simple PEA
- [ ] interpretace a validace biomarkeru v literature
- [ ] v obrazcich anotovat clustery

### For later
- [ ] prepocitat vsechno pro nove stazene datasety
  - [ ] BRCA
  - [ ] LGG
- [ ] pridat feature gnn
