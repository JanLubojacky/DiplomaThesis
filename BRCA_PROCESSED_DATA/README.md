# BRCA processed data
This folder contains processed BRCA data obtained from [xena browser](https://xenabrowser.net/datapages/?dataset=TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) at 20.3.2024

The data is preprocessed for the purpose of comparing different ML algorithms in the following way.

Only features for which all complete entires exist are kept and for mRNA and CNA variance filtering is applied, no feature selection is used as that should be applied during the training process and only on the training set, no normalization or min-max scaling is applied as that again should be based on the training set

**mrna.tsv**
- For mRNA low variance (var < 0.01) features are dropped

**mirna.tsv**
- no variance filtering is applied due to low number of complete mirna features
