# Notes
- A good point for discussion might be that when not enough samples is available it might be better to use a simpler model than a neural network
- What kinds of gene lengths should be used for normalizing? => lets use mean exon lenght
- Ensembl ids for rna-seq data are outdated, so retrieving gene lengths is not easy and would require significant amounts of manual annotations
- Could add one more task for mutations
  - SPL (splicing), EPI (epigenetic), WT (wild type), CTR (control)
- Could encode the strand as an additional feature, might be especially useful for feature based gnns where each node is a feature