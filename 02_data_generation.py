import numpy as np
import pandas as pd

np.random.seed(42)

states = ["sain_non_vaccine", "retabli", "contamine", "mort", "sain_vaccine", "infecte"]
n_states = len(states)

M1 = np.random.rand(n_states, n_states)
M1 = M1 / M1.sum(axis=1, keepdims=True)  

def generate_sequences(transition_matrix, n_sequences=150, seq_length=10):
   
    n_states = transition_matrix.shape[0]
    sequences = np.zeros((n_sequences, seq_length), dtype=int)
    
    for i in range(n_sequences):
        current_state = np.random.randint(0, n_states)
        sequences[i, 0] = current_state
        for t in range(1, seq_length):
            probs = transition_matrix[current_state]
            next_state = np.random.choice(n_states, p=probs)
            sequences[i, t] = next_state
            current_state = next_state
    
    return sequences

sequences_m1 = generate_sequences(M1, n_sequences=150, seq_length=10)
sequences_m1_names = [[states[idx] for idx in seq] for seq in sequences_m1]

df_sequences = pd.DataFrame(sequences_m1, columns=[f"Day_{i}" for i in range(10)])
df_sequences.to_csv("sequences_m1.csv", index=False)
