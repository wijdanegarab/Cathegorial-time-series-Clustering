import numpy as np
import pandas as pd

def dynamic_hamming_distance(seq1, seq2):
    return np.sum(seq1 != seq2)

def dynamic_hamming_matrix(sequences):
    n = sequences.shape[0]
    dist = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            d = dynamic_hamming_distance(sequences[i], sequences[j])
            dist[i, j] = dist[j, i] = d
    
    return dist

def cost_matrix(seq1, seq2, costs):
    m, n = len(seq1), len(seq2)
    matrix = np.zeros((m+1, n+1))
    
    for i in range(m+1):
        matrix[i, 0] = i * costs['indel']
    for j in range(n+1):
        matrix[0, j] = j * costs['indel']
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                cost = matrix[i-1, j-1]
            else:
                cost = min(
                    matrix[i-1, j-1] + costs['sub'],
                    matrix[i-1, j] + costs['indel'],
                    matrix[i, j-1] + costs['indel']
                )
            matrix[i, j] = cost
    
    return matrix[-1, -1]

def optimal_matching_matrix(sequences, costs):
    n = sequences.shape[0]
    dist = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            d = cost_matrix(sequences[i], sequences[j], costs)
            dist[i, j] = dist[j, i] = d
    
    return dist

sequences = pd.read_csv("sequences_m1.csv").values

costs = {'indel': 1, 'sub': 1}

dhd_dist = dynamic_hamming_matrix(sequences)
om_dist = optimal_matching_matrix(sequences, costs)

np.save("distance_matrix_dhd.npy", dhd_dist)
np.save("distance_matrix_om.npy", om_dist)
