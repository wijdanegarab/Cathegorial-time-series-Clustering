import numpy as np
import pandas as pd

np.random.seed(42)
STATES = ["sain_non_vaccine", "retabli", "contamine", "mort", "sain_vaccine", "infecte"]

def generate_matrix(dist_type, n_states):
    if dist_type == 'uniform':
        M = np.random.uniform(0, 1, (n_states, n_states))
    elif dist_type == 'gaussian':
        M = np.abs(np.random.standard_normal((n_states, n_states)))
    else:
        M = np.random.beta(2, 5, (n_states, n_states))
    
    return M / M.sum(axis=1, keepdims=True)

def generate_sequences(matrix, n_seq=150, seq_len=10):
    n_states = matrix.shape[0]
    sequences = np.zeros((n_seq, seq_len), dtype=int)
    sequences[:, 0] = np.random.randint(0, n_states, n_seq)
    
    for t in range(1, seq_len):
        for i in range(n_seq):
            sequences[i, t] = np.random.choice(n_states, p=matrix[sequences[i, t-1]])
    
    return sequences

for name, dist in [('m1', 'uniform'), ('m2', 'gaussian'), ('m3', 'beta')]:
    M = generate_matrix(dist, len(STATES))
    seq = generate_sequences(M)
    df = pd.DataFrame(seq, columns=[f"Day_{i}" for i in range(10)])
    df.to_csv(f"sequences_{name}.csv", index=False)
