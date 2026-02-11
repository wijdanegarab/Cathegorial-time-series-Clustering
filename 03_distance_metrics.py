import numpy as np
from itertools import permutations


def hamming_distance(seq1, seq2):
  
    if len(seq1) != len(seq2):
        raise ValueError("Les séquences doivent avoir la même longueur")
    
    distance = np.sum(seq1 != seq2)
    return distance


def hamming_distance_matrix(sequences):
    
    n_sequences = sequences.shape[0]
    dist_matrix = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(i+1, n_sequences):
            distance = hamming_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    return dist_matrix



def cost_matrix(seq1, seq2, indel_cost=1, substitution_cost=1):
    
    len1, len2 = len(seq1), len(seq2)
    matrix = np.zeros((len1 + 1, len2 + 1))
    
    # Initialiser première ligne et colonne (coûts d'insertion)
    for i in range(len1 + 1):
        matrix[i, 0] = i * indel_cost
    for j in range(len2 + 1):
        matrix[0, j] = j * indel_cost
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
           
            if seq1[i-1] == seq2[j-1]:
                substitution = matrix[i-1, j-1]  # Pas de coût si identique
            else:
                substitution = matrix[i-1, j-1] + substitution_cost
            
          
            deletion = matrix[i-1, j] + indel_cost
            
        
            insertion = matrix[i, j-1] + indel_cost
            
           
            matrix[i, j] = min(substitution, deletion, insertion)
    
    return matrix


def optimal_matching_distance(seq1, seq2, indel_cost=1, substitution_cost=1):
    
    matrix = cost_matrix(seq1, seq2, indel_cost, substitution_cost)
    return matrix[-1, -1]


def optimal_matching_matrix(sequences, indel_cost=1, substitution_cost=1):
   
    n_sequences = sequences.shape[0]
    dist_matrix = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(i+1, n_sequences):
            distance = optimal_matching_distance(
                sequences[i], 
                sequences[j], 
                indel_cost=indel_cost, 
                substitution_cost=substitution_cost
            )
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    return dist_matrix




def normalize_distance_matrix(dist_matrix):
 
    max_dist = np.max(dist_matrix)
    if max_dist == 0:
        return dist_matrix
    return dist_matrix / max_dist




if __name__ == "__main__":
    
    import pandas as pd
    
    sequences = pd.read_csv("sequences_m1.csv").values
    
    print("=" * 60)
    print("CALCUL DES DISTANCE MATRICES")
    print("=" * 60)
    
   
    print("\n1. HAMMING DISTANCE")
    print("-" * 60)
    dist_hamming = hamming_distance_matrix(sequences)
    print(f"Matrice Hamming shape: {dist_hamming.shape}")
    print(f"Min distance: {np.min(dist_hamming[np.nonzero(dist_hamming)]):.2f}")
    print(f"Max distance: {np.max(dist_hamming):.2f}")
    print(f"Mean distance: {np.mean(dist_hamming[np.nonzero(dist_hamming)]):.2f}")
    
   
    print("\n2. OPTIMAL MATCHING DISTANCE")
    print("-" * 60)
    dist_om = optimal_matching_matrix(sequences)
    print(f"Matrice OM shape: {dist_om.shape}")
    print(f"Min distance: {np.min(dist_om[np.nonzero(dist_om)]):.2f}")
    print(f"Max distance: {np.max(dist_om):.2f}")
    print(f"Mean distance: {np.mean(dist_om[np.nonzero(dist_om)]):.2f}")
    
   
    print("\n3. SAUVEGARDE")
    print("-" * 60)
    np.save("distance_matrix_hamming.npy", dist_hamming)
    np.save("distance_matrix_om.npy", dist_om)
    print("✓ Matrices sauvegardées:")
    print("  - distance_matrix_hamming.npy")
    print("  - distance_matrix_om.npy")
