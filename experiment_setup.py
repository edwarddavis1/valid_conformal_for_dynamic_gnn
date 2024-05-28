import numpy as np
import numba as nb


@nb.njit()
def make_iid(n, T=2, iid_prob=0.4, K=2):
    """
    Generates a series of nxn i.i.d. stochastic block model adjacency matrices

    Inputs
    n: number of nodes
    T: number of time points
    iid_prob: within-community edge probability for one of the communities (0.5 means the two are indisinguishable)
    K: number of communities

    Outputs
    As: Series of adjacency matrices (T, n, n)
    tau: Community assignments (n)
    changepoint: Middle of the series (variable exists to be in line with the other functions)
    """

    # Ensure equal community sizes
    if n % K != 0:
        raise ValueError("n must be divisible by the number of communities")

    if iid_prob > 1 or iid_prob < 0:
        raise ValueError("iid_prob must be between 0 and 1")

    tau = np.repeat(np.arange(0, K), int(n / K))
    np.random.shuffle(tau)

    # Generate B matrix
    if K == 2:
        B = np.array([[0.5, 0.5], [0.5, iid_prob]])
    else:
        move_amount = np.abs(iid_prob - 0.5)
        comm_probs = np.linspace(0.5 - move_amount, 0.5 + move_amount, K)
        B = np.ones((K, K)) * 0.5
        np.fill_diagonal(B, comm_probs)

    # If T is not even, round down T/2 for the changepoint
    if T % 2 == 0:
        changepoint = int(T / 2)
    else:
        changepoint = int(np.floor(T / 2))

    # Generate adjacency matrices
    As = np.zeros((T, n, n))
    for t in range(T):
        P_t = np.zeros((n, n))
        for i in range(n):
            P_t[:, i] = B[tau, tau[i]]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    return (As, tau, changepoint)
