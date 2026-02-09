import numpy as np
import pandas as pd

# Fixer seed pour reproductibilité
np.random.seed(42)

# ÉTAPE 1: Définir les états

states = ["sain_non_vaccine", "retabli", "contamine", "mort", "sain_vaccine", "infecte"]
n_states = len(states)
print(f"États disponibles: {states}")

# ÉTAPE 2: Créer matrice de transition M1 (Uniform)

M1 = np.random.rand(n_states, n_states)
M1 = M1 / M1.sum(axis=1, keepdims=True)  # Normaliser: chaque ligne somme à 1

print("\nMatrice M1 (Uniform):")
print(M1)

# ÉTAPE 3: Générer séquences avec M1

def generate_sequences(transition_matrix, n_sequences=150, seq_length=10):
    """
    Générer des séquences en utilisant une matrice de transition Markov
    
    Args:
        transition_matrix: matrice n_states x n_states
        n_sequences: nombre de séquences à générer
        seq_length: longueur de chaque séquence
    
    Returns:
        sequences: array (n_sequences, seq_length) avec indices d'états
    """
    n_states = transition_matrix.shape[0]
    sequences = np.zeros((n_sequences, seq_length), dtype=int)
    
    for i in range(n_sequences):
        # État initial aléatoire
        current_state = np.random.randint(0, n_states)
        sequences[i, 0] = current_state
        
        # Générer le reste de la séquence
        for t in range(1, seq_length):
            # Probabilités de transition depuis l'état courant
            probs = transition_matrix[current_state]
            # Choisir l'état suivant selon les probabilités
            next_state = np.random.choice(n_states, p=probs)
            sequences[i, t] = next_state
            current_state = next_state
    
    return sequences

# Générer séquences
sequences_m1 = generate_sequences(M1, n_sequences=150, seq_length=10)

print(f"\nForm des séquences: {sequences_m1.shape}")
print(f"Premières séquences (indices):\n{sequences_m1[:5]}")

# Convertir indices en noms d'états
sequences_m1_names = [[states[idx] for idx in seq] for seq in sequences_m1]
print(f"\nPremières séquences (noms):")
for i in range(3):
    print(f"Seq {i}: {' → '.join(sequences_m1_names[i])}")

# ÉTAPE 4: Sauvegarder les données

# Créer DataFrame pour une manipulation facile
df_sequences = pd.DataFrame(sequences_m1, columns=[f"Day_{i}" for i in range(10)])
df_sequences.to_csv("sequences_m1.csv", index=False)

print("\n✓ Données sauvegardées dans 'sequences_m1.csv'")
