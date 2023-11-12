# DiplomaThesis

## Datasets used
- BRCA
- data supplied by the IDA lab at CVUT FEL

## Baseline methods

## Graph neural networks
- GCN equations
	- **Aggregate information from neighbours**
		- in the easiest form this could be an average or we could use ...
		- a weighted sum $$h_{n(v)}=\sum_{u\in N(v)}w_{u,v}h_u$$
			- where $w_{u,v}$ is computed as a product of roots of inverse degrees $$w_{u,v}=\sqrt{\frac1{d_u}}\sqrt{\frac1{d_v}}$$
	- **Pass the aggregated vector through a MLP**
		- $h_{n+1(v)}=\sigma(W h_{n(v)})$
		- in each layer the same network is applied to all of the nodes
		- and usually we also use a different weight matrix
- **Relational GCNs**
	- relation is a triplet of $$\text{node type}\xrightarrow{\text{edge type}} \text{node type}$$
	- in RGCNs we have a different weight matrix for each of these triplets, then the update equation looks like $$h_i^{l+1}=\sigma\left(W_0^lh^l_i+\sum_{r\in R}\sum_{j\in N^r_i}\frac1{c_{ir}}W_r^lh_j^l\right)$$
	- and we can also introduce regularization
	- **Block diagonal matrix**
		- we allow the matrices $W_r$ to be block diagonal, to reduce the number of parameters (and allow only neighbouring embeddings to interact)
	- **Basis learning**
		- i.e. we put a cap on how. many base weight matrices $V_b$ we want per layer and compose the necessary matrices $W_r$ out of them via linear combination with learned coefficients, this is useful in case we have a heterogenous graph with many relations $$W_r^l=\sum_{b=1}^Ba_{rb}^lV_b^l$$
- **Attention GNNs**
	- in attention GNNs we modify the aggregation step, instead of a simple weight given by the degrees we compute attention weights $$h_{n(v)}=\sum_{u\in N(v)} \alpha(h_u,h_v) h_u$$
	- where $\alpha$ is a softmax of the attention scores $$\alpha(u,v)=\text{softmax}(a(h_u,h_v))$$
	- where $a$ can be computed as a
		- dot product
			- $a(h_u,h_v)=h_uh_v$
		- with a learnable set of parameters vector $a$ and matrix $W$ as
			- $a(h_u,h_v)=\text{LeakyReLU}(a^T\cdot[Wh_u||Wh_v])$
			- where the $[-||-]$ operator means concatenation of the two transformed vectors
	- which then gives the final equation for GATs as $$h_u=\sigma\left(\sum_{v\in N_u}\alpha_{uv}Wh_v\right)$$
		- where the weight matrix $W$ is the same weight matrix as the one that was used when computing the attention scores
	- we can also have multiple attention heads, meaning in each layer we have a set of weights $a^k,W^k$ for $k\in\{1,…,n\}$ for multiple attention heads
	- and we can concatenate / sum the output from each head into a new vector

## Baseline results

