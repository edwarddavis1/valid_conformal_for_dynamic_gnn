import os
import numpy as np
import spectral_embedding as se

K = 3
n = 100 * K
T = 8
pi = np.repeat(1 / K, K)

a = [0.08, 0.16]
Bs = 0.02 * np.ones((T, K, K))

T_list = [t for t in range(T)]
np.random.shuffle(T_list)

for t in range(T):
    for k in range(K):
        Bs[t, k, k] = a[(T_list[t] & (1 << k)) >> k]


As, Z = se.generate_SBM_dynamic(n, Bs, pi)

node_labels = np.tile(Z, T)

# Save As and node_labels
if not os.path.exists("processed_datasets/SBM"):
    os.makedirs("processed_datasets/SBM")

np.save("processed_datasets/SBM/As.npy", As)
np.save("processed_datasets/SBM/node_labels.npy", node_labels)
