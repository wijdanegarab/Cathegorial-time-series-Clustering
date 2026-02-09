import numpy as np
from itertools import permutations

# DISTANCE METRICS POUR SÉRIES TEMPORELLES CATÉGORIQUES

# 1. HAMMING DISTANCE (Dynamic Hamming Distance)


def hamming_distance(seq1, seq2):
    """
    Calculer la distance Hamming entre deux séquences
    
    Distance = nombre de positions où les deux séquences diffèrent
    
    Args:
        seq1, seq2: arrays de même longueur
    
    Returns:
        distance: nombre de positions différentes
    """
    if len(seq1) != len(seq2):
        raise ValueError("Les séquences doivent avoir la même longueur")
    
    distance = np.sum(seq1 != seq2)
    return distance


def hamming_distance_matrix(sequences):
    """
    Créer matrice de distances Hamming pour toutes les paires de séquences
    
    Args:
        sequences: array (n_sequences, seq_length)
    
    Returns:
        dist_matrix: array (n_sequences, n_sequences) symétrique
    """
    n_sequences = sequences.shape[0]
    dist_matrix = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(i+1, n_sequences):
            distance = hamming_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    return dist_matrix



# 2. OPTIMAL MATCHING DISTANCE (OM)


def cost_matrix(seq1, seq2, indel_cost=1, substitution_cost=1):
    """
    Créer matrice de coûts pour Optimal Matching
    
    Args:
        seq1, seq2: séquences à comparer
        indel_cost: coût de insertion/deletion
        substitution_cost: coût de substitution
    
    Returns:
        cost_matrix: matrice (len(seq1)+1, len(seq2)+1)
    """
    len1, len2 = len(seq1), len(seq2)
    matrix = np.zeros((len1 + 1, len2 + 1))
    
    # Initialiser première ligne et colonne (coûts d'insertion)
    for i in range(len1 + 1):
        matrix[i, 0] = i * indel_cost
    for j in range(len2 + 1):
        matrix[0, j] = j * indel_cost
    
    # Remplir la matrice avec programmation dynamique
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Coût si on substitue
            if seq1[i-1] == seq2[j-1]:
                substitution = matrix[i-1, j-1]  # Pas de coût si identique
            else:
                substitution = matrix[i-1, j-1] + substitution_cost
            
            # Coût si on delete
            deletion = matrix[i-1, j] + indel_cost
            
            # Coût si on insert
            insertion = matrix[i, j-1] + indel_cost
            
            # Prendre le minimum
            matrix[i, j] = min(substitution, deletion, insertion)
    
    return matrix


def optimal_matching_distance(seq1, seq2, indel_cost=1, substitution_cost=1):
    """
    Calculer Optimal Matching distance entre deux séquences
    
    C'est le coût minimum pour transformer seq1 en seq2
    via insertions, deletions, substitutions
    
    Args:
        seq1, seq2: séquences à comparer
        indel_cost: coût de insertion/deletion
        substitution_cost: coût de substitution
    
    Returns:
        distance: coût minimum
    """
    matrix = cost_matrix(seq1, seq2, indel_cost, substitution_cost)
    return matrix[-1, -1]


def optimal_matching_matrix(sequences, indel_cost=1, substitution_cost=1):
    """
    Créer matrice de distances Optimal Matching pour toutes les paires
    
    Args:
        sequences: array (n_sequences, seq_length)
        indel_cost: coût de insertion/deletion
        substitution_cost: coût de substitution
    
    Returns:
        dist_matrix: array (n_sequences, n_sequences) symétrique
    """
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

# 3. NORMALISER LES DISTANCES


def normalize_distance_matrix(dist_matrix):
    """
    Normaliser la matrice de distances au range [0, 1]
    
    Args:
        dist_matrix: matrice de distances
    
    Returns:
        normalized: matrice normalisée
    """
    max_dist = np.max(dist_matrix)
    if max_dist == 0:
        return dist_matrix
    return dist_matrix / max_dist


# TEST


if __name__ == "__main__":
    # Importer les données générées
    import pandas as pd
    
    sequences = pd.read_csv("sequences_m1.csv").values
    
    print("=" * 60)
    print("CALCUL DES DISTANCE MATRICES")
    print("=" * 60)
    
    # Hamming Distance
    print("\n1. HAMMING DISTANCE")
    print("-" * 60)
    dist_hamming = hamming_distance_matrix(sequences)
    print(f"Matrice Hamming shape: {dist_hamming.shape}")
    print(f"Min distance: {np.min(dist_hamming[np.nonzero(dist_hamming)]):.2f}")
    print(f"Max distance: {np.max(dist_hamming):.2f}")
    print(f"Mean distance: {np.mean(dist_hamming[np.nonzero(dist_hamming)]):.2f}")
    
    # Optimal Matching
    print("\n2. OPTIMAL MATCHING DISTANCE")
    print("-" * 60)
    dist_om = optimal_matching_matrix(sequences)
    print(f"Matrice OM shape: {dist_om.shape}")
    print(f"Min distance: {np.min(dist_om[np.nonzero(dist_om)]):.2f}")
    print(f"Max distance: {np.max(dist_om):.2f}")
    print(f"Mean distance: {np.mean(dist_om[np.nonzero(dist_om)]):.2f}")
    
    # Sauvegarder
    print("\n3. SAUVEGARDE")
    print("-" * 60)
    np.save("distance_matrix_hamming.npy", dist_hamming)
    np.save("distance_matrix_om.npy", dist_om)
    print("✓ Matrices sauvegardées:")
    print("  - distance_matrix_hamming.npy")
    print("  - distance_matrix_om.npy")
