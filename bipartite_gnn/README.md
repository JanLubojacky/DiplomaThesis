# bipartite gnn architecture
- input is a bipartite graph with patients on one side and genes / mirnas on the other side
- patient vertices contain information for multiple omic channels
- feature vertices contain a vector of ones for start
  - we have different types of feature vectors for genes, mirnas - feature - sample, etc.
  - we add edges between samples and features based on thresholding over/under expression of the features across the different classes
    - this could be also used for some simple feature selection? select only genes with differential expression across the different classes
  - finally we add edges between samples based on feature interactions such as gene-gene interaction or mirna-gene interactions
- any longest shortest path in this graph is at most 3 edges `sample -> feature -> feature -> sample`, this means that we need 3 layers, with 2 we wouldn't be utilizing the feature-feature interactions as the information trough them wouldnt propagate to the samples and 4 would cause oversmoothing
- it helps to have all the features of the same dimensions, we can use a simple linear layer at the start to scale all the features to the same dimension like a 100 or 50? (this is a hyperparameter we should tune) this could be an autoencoder but it will likely be better to have this learned
- we will use a different weight for each modality like in rgcns
- we also need to make sure that each feature node receives the correct feature vector, i.e. each sample node will contain several channels of information, (mrna expression, mirna expression, dna methylation, cna)
- question is, should we split the feature vertices for mrna expression and dna methylation / cna if they are for the same gene? answer probably yes, each might have different edges with samples, then we have to make sure that each channel from the sample vertices gets propagated to the correct vertices
- we will also use attention, it usually has better performance than classical convolution
- include pooling?

## Initial analysis
- lets use rna-seq, dna methylation and mirna-seq data

## Alternative model
- sample vertices start empty (vector of zeros), and input vertices contain info for all inputs
- this has problems that I am not sure can be overcome
