import numpy as np
import numba as nb


@nb.njit()
def make_temporal_simple(n, T=2, move_prob=0.53, K=2):
    """
    Generates a series of nxn adjacency matrices where one of two communities has constant behaviour
    over two time points and the other moves (when move_prob != 0.53).

    Inputs
    n: number of nodes
    T: number of time points
    move_prob: within-community edge probability for the MOVING-COMMUNITY for the second time point (initially 0.5)
    K: number of communities

    Outputs
    As: Series of adjacency matrices (T, n, n)
    tau: Community assignments (n)
    changepoint: Time point at which the MOVING-COMMUNITY changes
    """

    # Ensure equal community sizes
    if n % K != 0:
        raise ValueError("n must be divisible by the number of communities")

    tau = np.repeat(np.arange(0, K), int(n / K))
    np.random.shuffle(tau)

    # Generate B matrices
    B_list = np.zeros((2, K, K))

    if K == 2:
        B_list[0] = np.array([[0.5, 0.2], [0.2, 0.5]])
        B_list[1] = np.array([[0.5, 0.2], [0.2, move_prob]])
    else:
        move_amount = np.abs(move_prob - 0.5)
        comm_probs = np.linspace(0.3, 0.9, K)
        B_for_time = np.ones((K, K)) * 0.2
        np.fill_diagonal(B_for_time, comm_probs)
        B_list[0] = B_for_time
        B_for_change = B_for_time.copy()
        B_for_change[1, 1] = B_for_change[1, 1] + move_amount
        B_list[1] = B_for_change

    # If T is not even, round down T/2 for the changepoint
    if T % 2 == 0:
        changepoint = int(T / 2)
    else:
        changepoint = int(np.floor(T / 2))

    # Generate adjacency matrices
    As = np.zeros((T, n, n))
    for t in range(0, changepoint):
        P_t = np.zeros((n, n))
        for i in range(n):
            P_t[:, i] = B_list[0][tau, tau[i]]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    for t in range(changepoint, T):
        P_t = np.zeros((n, n))
        for i in range(n):
            P_t[:, i] = B_list[1][tau, tau[i]]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    return (As, tau, changepoint)


@nb.njit()
def get_power_weights(n, max_exp_deg, w_param=6, beta=2.5):
    """Generate weights according to a power law distribution with 2 < beta < 3
    beta should be at least 2.5 for ASE
    https://arxiv.org/pdf/0810.1355.pdf
    """
    alpha = (beta - 2) / (beta - 1)
    c = alpha * w_param * np.power(n, 1 / (beta - 1))
    i0 = n * np.power(alpha * (w_param / max_exp_deg), beta - 1)
    iterarr = np.arange(0, n) + i0
    w = np.array([c * np.power(i, -1 / (beta - 1)) for i in iterarr])
    w_sum = np.sum(w)

    if w.max() ** 2 >= np.sum(w):
        print(
            "WARNING: Weights distribution for the power experiment is not sufficient. Try increasing n."
        )

    W = (w.reshape((n, 1)) @ w.reshape((n, 1)).T) / w_sum
    return W


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


# @nb.njit()
def make_iid_with_allocations(n, T=2, iid_prob=0.4, K=2, allocation_probs=None):
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

    if allocation_probs is None:
        tau = np.repeat(np.arange(0, K), int(n / K))
    else:
        tau = np.random.choice(np.arange(0, K), n, p=allocation_probs)

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


@nb.njit()
def make_merge(n, T=2):
    """
    Generates a series of nxn adjacency matrices where two initially separate communities merge

    Inputs
    n: number of nodes
    T: number of time points

    Outputs
    As: Series of adjacency matrices (T, n, n)
    tau: Community assignments (n)
    changepoint: Time point at which the communities merge
    """

    K = 2
    if n % K != 0:
        raise ValueError("n must be divisible by the number of communities")

    tau = np.repeat(np.arange(0, K), int(n / K))
    np.random.shuffle(tau)

    # Generate B matrix
    B_list = np.zeros((2, K, K))
    B_list[0] = np.array([[0.9, 0.2], [0.2, 0.1]])
    B_list[1] = np.array([[0.5, 0.5], [0.5, 0.5]])

    # If T is not even, round down T/2 for the changepoint
    if T % 2 == 0:
        changepoint = int(T / 2)
    else:
        changepoint = int(np.floor(T / 2))

    # Generate adjacency matrices
    As = np.zeros((T, n, n))
    for t in range(0, changepoint):
        P_t = np.zeros((n, n))
        for i in range(n):
            P_t[:, i] = B_list[0][tau, tau[i]]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    for t in range(changepoint, T):
        P_t = np.zeros((n, n))
        for i in range(n):
            P_t[:, i] = B_list[1][tau, tau[i]]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    return (As, tau, changepoint)


def make_experiment(experiment, n, T, move_prob=0.53, power_move_prob=0.97):
    """
    Generates a dynamic network according to the experiment specified.

    Inputs:
    experiment: String specifying the experiment to run
    n: Number of nodes
    T: Number of time points
    move_prob: Amount of movement in the moving experiment. 0.5 = no move.
    power_move_prob: Amount of movement in the POWER-MOVING experiment. 1 = no move.

    Outputs:
    As: Adjacency matrix series
    Tau: Community labels
    clust_to_check: Community to test
    changepoint: Separation point of the two embedding sets to test
    """

    clust_to_check = 0
    if experiment.upper() == "STATIC" or experiment.upper() == "STATIC-SPATIAL":
        As, tau, changepoint = make_iid(n, T)
    elif experiment.upper() == "MOVING-COMMUNITY":
        As, tau, changepoint = make_temporal_simple(n, T, move_prob=move_prob)
        clust_to_check = 1
    elif experiment.upper() == "MOVING-STATIC-COMMUNITY":
        As, tau, changepoint = make_temporal_simple(n, T, move_prob=move_prob)
    elif experiment.upper() == "POWER-MOVING":
        As = make_iid_close_power(
            n, T, max_exp_deg=6, change_prob=power_move_prob, w_param=6
        )
        tau = np.zeros((n,))
        changepoint = 1
    elif experiment.upper() == "POWER-STATIC":
        As = make_iid_close_power(n, T, max_exp_deg=6, change_prob=1, w_param=6)
        tau = np.zeros((n,))
        changepoint = 1
    elif experiment.upper() == "MERGE":
        As, tau, changepoint = make_merge(n, T)
    else:
        raise ValueError(
            "{} is not a recognised system\n- Please select from:\n\t> STATIC\n\t> MOVING-COMMUNITY\n\t> MOVING-STATIC-COMMUNITY\n\t> POWER-MOVING\n\t> POWER-STATIC\n\t> MERGE\n\t> STATIC-SPATIAL\n".format(
                experiment
            )
        )

    return (As, tau, clust_to_check, changepoint)


def get_B_for_exp(experiment, T=2, move_prob=0.53, K=2, power_move_prob=0.97):
    """
    Returns the SBM matrix for each system.
    """
    if experiment.upper() == "STATIC" or experiment.upper() == "STATIC-SPATIAL":
        B = np.array([[[0.8, 0.2], [0.2, 0.4]]] * T)
    elif experiment.upper() in ["MOVING-COMMUNITY", "MOVING-STATIC-COMMUNITY"]:
        B = np.zeros((T, K, K))
        B[0 : int(T / 2)] = np.array([[[0.5, 0.2], [0.2, 0.5]]] * int(T / 2))
        B[int(T / 2) :] = np.array([[[0.5, 0.2], [0.2, move_prob]]] * int(T / 2))
    elif experiment.upper() in ["MERGE"]:
        B = np.zeros((T, K, K))
        B[0 : int(T / 2)] = np.array([[[0.9, 0.2], [0.2, 0.1]]] * int(T / 2))
        B[int(T / 2) :] = np.array([[[0.5, 0.5], [0.5, 0.5]]] * int(T / 2))
    elif experiment.upper() in ["POWER-MOVING"]:
        B = np.zeros((T, K, K))

        B[0 : int(T / 2)] = np.array([[[1, 1], [1, 1]]] * int(T / 2))
        B[int(T / 2) :] = np.array(
            [[[power_move_prob, power_move_prob], [power_move_prob, power_move_prob]]]
            * int(T / 2)
        )

    elif experiment.upper() in ["POWER-STATIC"]:
        B = np.zeros((T, K, K))
        B[0 : int(T / 2)] = np.array([[[1, 1], [1, 1]]] * int(T / 2))
        B[int(T / 2) :] = np.array([[[1, 1], [1, 1]]] * int(T / 2))
    else:
        raise Exception(
            "{} is not a recognised system\n- Please select from:\n\t> STATIC\n\t> MOVING-COMMUNITY\n\t> MOVING-STATIC-COMMUNITY\n\t> POWER-MOVING\n\t> POWER STABLE\n\t> MERGE".format(
                experiment
            )
        )

    return B


@nb.njit()
def form_omni_matrix(As, n, T):
    """
    Forms the embedding matrix for the omnibus embedding
    """
    A = np.zeros((T * n, T * n))

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (As[t1] + As[t2]) / 2

    return A


def get_embedding_dimension(P_list, method):
    """
    Select the embedding dimension to be the rank of the method's embedding matrix
    """
    if (
        method.upper() == "ISE"
        or method.upper() == "ISE PROCRUSTES"
        or method.upper() == "INDEPENDENT NODE2VEC"
    ):
        d = []
        for t in range(len(P_list)):
            d.append(np.linalg.matrix_rank(P_list[t]))

    elif (
        method.upper() == "UASE"
        or method.upper() == "ULSE"
        or method.upper() == "URLSE"
        or method.upper() == "UNFOLDED NODE2VEC"
        or method.upper() == "DYNAMIC SKIP GRAM"
        or method.upper() == "GLODYNE"
    ):
        P = np.column_stack(P_list)
        d = np.linalg.matrix_rank(P)

    elif method.upper() == "OMNI":
        T = len(P_list)
        n = P_list[0].shape[0]
        P_omni = form_omni_matrix(P_list, n, T)

        d = np.linalg.matrix_rank(P_omni)

    else:
        raise Exception(
            "{} is not a recognised embedding method\n- Please select from:\n\t> ISE\n\t> ISE PROCRUSTES\n\t> OMNI\n\t> UASE\n\t> ULSE\n\t> URLSE\n\t> INDEPENDENT NODE2VEC\n\t> UNFOLDED NODE2VEC\n\t> GLODYNE\n".format(
                method
            )
        )

    return d


@nb.njit()
def make_iid_close_power(n, T=2, max_exp_deg=6, beta=2.5, change_prob=0.9, w_param=6):
    """
    Generates a series of nxn i.i.d. stochastic block model adjacency matrices with a power law distribution of weights

    Inputs
    n: number of nodes
    T: number of time steps
    max_exp_deg, beta, w_param: parameters for the power law distribution
    change_prob: intercommunity edge probabilities (before applying power law distribution)

    Outputs
    As: list of adjacency matrices
    """

    # Generate weights according to a power law distribution with 2 < beta < 3
    # https://arxiv.org/pdf/0810.1355.pdf

    W = get_power_weights(n=n, max_exp_deg=max_exp_deg, w_param=w_param, beta=beta)

    # If T is not even, round down T/2 for the changepoint
    if T % 2 == 0:
        changepoint = int(T / 2)
    else:
        changepoint = int(np.floor(T / 2))

    prob_for_times = [1, change_prob]

    # Generate adjacency matrices
    As = np.zeros((T, n, n))
    for t in range(0, changepoint):
        P_t = W * prob_for_times[0]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    for t in range(changepoint, T):
        P_t = W * prob_for_times[1]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t

    return As


def make_inhomogeneous_rg(P):
    n = P.shape[0]
    A = np.random.uniform(0, 1, n**2).reshape((n, n)) < P
    A = A.astype(float)

    return A


def sbm_from_B(n, Bs, return_p=False):
    T = len(Bs)
    K = Bs[0].shape[0]

    if n % K != 0:
        raise Exception("n must be divisible by K")

    tau = np.repeat(np.arange(0, K), int(n / K))
    np.random.shuffle(tau)

    # Generate adjacency matrices
    As = np.zeros((T, n, n))
    Ps = np.zeros((T, n, n))
    for t in range(T):
        P_t = np.zeros((n, n))
        for i in range(n):
            P_t[:, i] = Bs[t][tau, tau[i]]

        A_t = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_t
        As[t] = A_t
        Ps[t] = P_t

    if not return_p:
        return (As, tau)
    else:
        return (As, tau, Ps)
