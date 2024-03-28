# Schuzka 28.3.2024 notes
- **Building graph based on differential expression**
  - v bipartitnim grafu jsou uzly pacientu a genu propojeny na zaklade diferencialni exprese genu, posledne jsme diskutovali o tom ze rozhodovat o difirencialni expresi na zaklade odchylky od prumeru neni idealni, jednak exprese pochazi z negativniho binomialniho rozdeleni, druhak dulezita je i zmena ve varianci exprese a ne jen v prumeru
  - nasel jsem dve metody, ktere umi udelat takove analyzy, plus paper ktery je porovnava `ADD PAPERS`

- **Biomarker discovery progress**
  - **Learning important features**
    - jedna z veci ktera je v modelech stejne potreba udelat je projekce ruznych feature vektoru na stejnou dimenzi (typicky miRNA je velmi malo, mRNA naopak hodne), protoze je pak chceme mit v jednom grafu a michat je mezi sebou pres interakcni hrany
    - toto prida par procent na metrikach, ale hlavne jsou pak vysledky modelu mnohem konzistentnejsi a stabilnejsi
    - napad je nechat hodne featur ve vstupnim vektrou a pouzit k projekci linearni vrstvu se silnou L1 regularizaci, coz donuti model vytvit nizkodimenzionalni reprezentaci pouze z tech nejdulezitejsich features, vahy v teto projekcni vrstve pak muzeme pouzit k vyjadreni dulezitosti jednotlivych features
    - tento pristup ma sve opodstatneni v literature `ADD PAPERS`

- **New benchmarks**

# Schuzka 21.3.2024 notes
- **Architecture notes**
  - bipartite model construction offers a form of additional feature selection, where non-informative nodes will become unconnected
  - a 3 layer relational architecture with GAT layers
  - summary
    - feature projection module to align the dimension of features
    - interactions

- **Biomarker discovery**
  - Using backward feature removal
    - this is done in MOGONET, for all samples in test set run testing feature_num times, each time setting a feature to zero for all samples and observe the drop in performance
    - in the bipartite architecture this approach can be extended by isolating the corresponding feature nodes in the network
  - When we use GAT layers each node gives attention weights to its neighbours, we can then look at these weights for each sample and see which neighbours (features in the bipartite graph) it attends to the most

- **Future**
  - extend to more datasets and create tables with results and comparisions vs other architectures
  - implement feature importance mechanisms
  - current experiments hopefully mostly finished by the end of march


# Results, disease task

| Method | Accuracy | F1 macro | F1 weighted |
| --- | --- | --- | --- |
| KNN | 0.85 +/- 0.05 | 0.64 +/- 0.14 | 0.82 +/- 0.06 |
| LIN SVM | 0.80 +/- 0.08 | 0.65 +/- 0.15 | 0.80 +/- 0.08 |
| RBF SVM | 0.85 +/- 0.06 | 0.68 +/- 0.16 | 0.83 +/- 0.07 |
| XGBoost | 0.95 +/- 0.03 | 0.90 +/- 0.07 | 0.94 +/- 0.04 |
| MLP | 0.87 | 0.66 | 0.84 |
| MogonetGCN | 0.87 | 0.75 | 0.86 |
| BiGNN | 0.91 | 0.85 | 0.91 |
| MogonetStyleGAT | 0.96 | 0.92 | 0.95 |

# Results, risk task

| Method | Accuracy | F1 macro | F1 weighted |
| --- | --- | --- | --- |
| KNN | 0.50 +/- 0.09 | 0.47 +/- 0.10 | 0.47 +/- 0.10 |
| LIN SVM | 0.53 +/- 0.13 | 0.52 +/- 0.13 | 0.52 +/- 0.13 |
| RBF SVM | 0.52 +/- 0.12 | 0.50 +/- 0.13 | 0.51 +/- 0.13 |
| XGBoost | 0.64 +/- 0.06 | 0.65 +/- 0.07 | 0.64 +/- 0.07 |
| MogonetGCN | 0.70 | 0.69 | 0.70 |
| MogonetGAT | 0.74 | 0.75 | 0.74 |
| BiGNN | 0.74 | 0.74 | 0.74 |

# Results, mutation task

| Method | Accuracy | F1 macro | F1 weighted |
| --- | --- | --- | --- |
| KNN | 0.55 +/- 0.10 | 0.41 +/- 0.12 | 0.55 +/- 0.09 |
| LIN SVM | 0.57 +/- 0.09 | 0.47 +/- 0.09 | 0.58 +/- 0.09 |
| RBF SVM | 0.64 +/- 0.09 | 0.47 +/- 0.12 | 0.61 +/- 0.09 |
| XGBoost | 0.71 +/- 0.03 | 0.49 +/- 0.08 | 0.65 +/- 0.05 |
| MLP | 0.65 | 0.36 | 0.57 |
| MogonetGCN | 0.70 | 0.40 | 0.60 |
| MogonetGAN | 0.74 | 0.60 | 0.71 |
| BiGNN | 0.74 | 0.61 | 0.71 |
