# Schuzka 21.3.2024 notes
- **Architecture notes**
  - bipartite model construction offers a form of additional feature selection, where non-informative nodes will become unconnected
  - the best configuration seems to be a 3 layer relational architecture
  - summary
    - feature projection module
    - 

- **Biomarker discovery**
  - Using backward feature removal
    - this is done in MOGONET, for all samples in test set run testing feature_num times, each time setting a feature to zero for all samples and observe the drop in performance
    - in the bipartite architecture this approach can be extended by isolating the corresponding feature nodes in the network
  - When we use GAT layers each node gives attention weights to its neighbours, we can then look at these weights for each sample and see which neighbours (features in the bipartite graph) it attends to the most
