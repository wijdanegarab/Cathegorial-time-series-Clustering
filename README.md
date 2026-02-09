# Clustering of Categorical Time Series

## Overview
Python implementation of unsupervised clustering algorithms for categorical time series data. Based on INSERM internship research project on epidemiological disease progression modeling.

## Project Description

This project implements and compares multiple clustering approaches for categorical temporal sequences:

### Algorithms
- **K-Medoids Clustering**: Partition-based clustering using medoid representatives
- **Hierarchical Clustering**: Agglomerative clustering with Ward's linkage method

### Distance Metrics
- **Dynamic Hamming Distance (DHD)**: Position-wise difference count
- **Optimal Matching (OM)**: Minimum-cost sequence transformation using dynamic programming

### Evaluation Metrics
- **Adjusted Rand Index (ARI)**: Agreement between predicted and true labels
- **Silhouette Score**: Intrinsic clustering quality assessment

## Data

- **Source**: Generated using Markov chains with different transition matrix distributions (random)
- **Format**: 150 sequences × 10 time steps
- **States**: 6 categorical states
  - `sain_non_vaccine` (healthy unvaccinated)
  - `retabli` (recovered)
  - `contamine` (contaminated)
  - `mort` (deceased)
  - `sain_vaccine` (healthy vaccinated)
  - `infecte` (infected)

### Data Generation
Three transition matrices generated with different distributions:
- M1: Uniform distribution
- M2: Absolute Gaussian distribution
- M3: Beta distribution

## Project Structure
```
.

├── requirements.txt                   # Python dependencies
├── data_generation.py                 # Generate synthetic sequences
├── distance_metrics.py                # Distance calculations
├── clustering.py                      # Clustering algorithms
├── evaluation.py                      # Evaluation metrics
├── sequences_m1.csv                   # Generated sequences
├── distance_matrix_hamming.npy        # Hamming distance matrix
├── distance_matrix_om.npy             # Optimal matching distance matrix
├── dendrogram_hamming.png             # Hierarchical clustering visualization
├── dendrogram_om.png                  # Hierarchical clustering visualization
├── clusters_hc_hamming.png            # 2D cluster visualization
├── clusters_hc_om.png                 # 2D cluster visualization
├── silhouette_hc_hamming.png          # Silhouette plot
├── silhouette_hc_om.png               # Silhouette plot
└── evaluation_results.csv             # Summary of results
```

## Key Findings

### Best Performing Method
**Hierarchical Clustering + Optimal Matching**
- ARI: 0.1241
- Silhouette Score: 0.0672

### Ranking
1. Hierarchical (OM): ARI=0.124, Silhouette=0.067
2. K-Medoids (OM): ARI=0.094, Silhouette=0.054
3. Hierarchical (DHD): ARI=0.005, Silhouette=0.018
4. K-Medoids (DHD): ARI=0.017, Silhouette=0.040

**Insight**: Optimal Matching distance captures sequence similarity better than Hamming distance for this data.

## Requirements
```
see requirements.txt, libraries need to be installed
```


## Usage

### 1. Generate synthetic data
```
python data_generation.py
```
Generates 150 sequences using Markov chains and saves to `sequences_m1.csv`

### 2. Calculate distance matrices
```
python distance_metrics.py
```
Computes Hamming and Optimal Matching distances. Outputs:
- `distance_matrix_hamming.npy`
- `distance_matrix_om.npy`

### 3. Perform clustering
```
python clustering.py
```
Applies K-Medoids and Hierarchical clustering. Generates:
- Cluster labels (npy files)
- Dendrograms
- 2D cluster visualizations

### 4. Evaluate results
```
python evaluation.py
```
Computes ARI and Silhouette scores. Generates:
- `evaluation_results.csv`
- Silhouette plots

### Run all steps
```
python data_generation.py && python distance_metrics.py && python clustering.py && python evaluation.py
```

## Methods Explanation

### Distance Metrics

**Hamming Distance**
```
DHD(seq1, seq2) = Σ I(seq1[i] ≠ seq2[i])
```
Simple count of positions where sequences differ.

**Optimal Matching**
```
OM(seq1, seq2) = min cost to transform seq1 into seq2
Operations: insertion, deletion, substitution
```
Uses dynamic programming to find minimum-cost alignment.

### Clustering Algorithms

**K-Medoids**
- Partition-based approach
- Robust to outliers (uses representative points, not centroids)
- Requires pre-specified k

**Hierarchical Clustering (Agglomerative)**
- Bottom-up approach building dendrogram
- No need to specify k beforehand
- Ward linkage minimizes within-cluster variance

### Evaluation

**Adjusted Rand Index**
- Compares predicted vs true labels
- Adjusts for chance agreement
- Range: [-1, 1] (1=perfect)

**Silhouette Score**
- Measures how similar point is to its own cluster vs others
- Range: [-1, 1] (1=well-clustered)
- Works without ground truth labels

## Results & Interpretation

The analysis reveals:

1. **Optimal Matching > Hamming Distance**: OM captures sequential similarity better because it accounts for sequence alignment, not just position-wise differences.

2. **Hierarchical > K-Medoids**: Hierarchical clustering with Ward linkage performs better, likely due to better distance-based merging criteria.

3. **Low Absolute Scores**: Lower ARI/Silhouette values suggest the data has inherent ambiguity or overlapping clusters - this is realistic for epidemiological data.

4. **Practical Insight**: For disease progression modeling, temporal alignment (OM) matters more than exact state matching (Hamming).

## Applications

This methodology is applicable to:
- **Epidemiology**: Disease progression pattern recognition
- **Finance**: Market regime clustering (regime switching models)
- **Sequence Analysis**: Pattern detection in categorical time series
- **Healthcare**: Patient trajectory clustering



## Author

Wijdane GARAB  
INSERM Internship Project  
Paris Cité University

## References

- Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences.
- Batagelj, V., & Bren, M. (1995). Comparing resemblance measures.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.

## License

Bio-Informatics Engineering

## Acknowledgments

Thanks to INSERM supervisors François PETIT and Ottavio KHALIFA for the research guidance.

