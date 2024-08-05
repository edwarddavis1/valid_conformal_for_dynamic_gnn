# import networkx as nx
from scipy.spatial import distance
from scipy import stats
import numpy as np
import pandas as pd
from scipy import sparse
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numba as nb

# from src.libne.DynWalks import DynWalks
import networkx as nx


# def GloDyNE(As, d, limit=0.1, num_walks=20, walklen=30, window=10, scheme=4, sparse_matrix=False):
#     T = len(As)
#     n = As[0].shape[0]
#     graphs = []
#     for t in range(T):
#         if sparse_matrix:
#             graphs.append(nx.from_scipy_sparse_matrix(As[t]))
#         else:
#             graphs.append(nx.from_numpy_array(As[t]))

#     # Limit is alpha in the paper
#     model = DynWalks(G_dynamic=graphs, limit=limit, num_walks=num_walks, walk_length=walklen, window=window,
#                      emb_dim=d, negative=5, workers=32, seed=2019, scheme=scheme)
#     ya_raw = model.sampling_traning()

#     ya = np.zeros((n*T, d))
#     for t in range(T):
#         for i in range(n):
#             ya[i + n*t, :] = ya_raw[t][i]

#     return(ya)


def SSE(As, d, flat=True, procrustes=False, consistent_orientation=True):
    """Computes separate spectral embeddings for each adjacency snapshot"""

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    if not isinstance(d, list):
        d_list = [d] * T
    else:
        d_list = d

    # Compute embeddings for each time
    YA_list = []
    for t in range(T):
        UA, SA, _ = sparse.linalg.svds(As[t], d_list[t])
        idx = SA.argsort()[::-1]
        UA = UA[:, idx]
        SA = SA[idx]

        # Make sure the eigenvector orientation choice is consistent
        if consistent_orientation:
            sum_of_ev = np.sum(UA, axis=0)
            for i in range(sum_of_ev.shape[0]):
                if sum_of_ev[i] < 0:
                    UA[:, i] = -1 * UA[:, i]

        embed = UA @ np.diag(np.sqrt(SA))

        if embed.shape[1] < max(d_list):
            empty_cols = np.zeros((n, max(d_list) - embed.shape[1]))
            embed = np.column_stack([embed, empty_cols])

        if procrustes and t > 0:
            # Align with previous embedding
            w1, s, w2t = np.linalg.svd(previous_embed.T @ embed)
            w2 = w2t.T
            w = w1 @ w2.T
            embed_rot = embed @ w.T
            embed = embed_rot

        YA_list.append(embed)
        previous_embed = embed

    # Format the output
    if flat:
        YA = np.row_stack(YA_list)
    else:
        YA = np.zeros((T, n, max(d_list)))
        for t in range(T):
            YA[t, :, :] = YA_list[t]

    return YA


def procrust_align(previous_embed, embed):
    w1, _, w2t = np.linalg.svd(previous_embed.T @ embed)
    w2 = w2t.T
    w = w1 @ w2.T
    embed_rot = embed @ w.T

    return embed_rot


def single_spectral(A, d, seed=None):
    if seed is not None:
        UA, SA, VAt = sparse.linalg.svds(A, d, random_state=seed)
    else:
        UA, SA, VAt = sparse.linalg.svds(A, d)

    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    XA = UA @ np.diag(np.sqrt(SA))
    return XA


def sparse_stack(As):
    A = As[0]
    T = len(As)
    for t in range(1, T):
        A = sparse.hstack((A, As[t]))
    return A


def UASE(As, d, flat=True, sparse_matrix=False, return_left=False):
    """Computes the unfolded adjacency spectral embedding"""
    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    # SVD spectral embedding
    UA, SA, VAt = sparse.linalg.svds(A, d)
    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    YA_flat = VA @ np.diag(np.sqrt(SA))
    XA = UA @ np.diag(np.sqrt(SA))
    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t, :, :] = YA_flat[n * t : n * (t + 1), :]

    if not return_left:
        return YA
    else:
        return XA, YA


# def independent_n2v(As, d, p=1, q=1, flat=True, num_walks=20, window=10, walklen=30, verbose=False, sparse_matrix=False):
#     """Computes an independent node2vec embedding for each adjacency snapshot"""

#     # Assume fixed n over time
#     n = As[0].shape[0]
#     T = len(As)

#     if not isinstance(d, list):
#         d_list = [d] * T
#     else:
#         d_list = d

#     YA_list = []
#     for t in range(T):
#         n2v_obj = nodevectors.Node2Vec(
#             n_components=d_list[t],
#             epochs=num_walks,
#             walklen=walklen,
#             return_weight=1/p,
#             neighbor_weight=1/q,
#             verbose=verbose,
#             w2vparams={"window": window, "negative": 5, "iter": 10,
#                        "batch_words": 128}
#         )
#         n2v_embed = n2v_obj.fit_transform(As[t])

#         if n2v_embed.shape[1] < max(d_list):
#             empty_cols = np.zeros((n, max(d_list) - n2v_embed.shape[1]))
#             n2v_embed = np.column_stack([n2v_embed, empty_cols])

#         YA_list.append(n2v_embed)

#     if flat:
#         YA = np.row_stack(YA_list)
#     else:
#         YA = np.zeros((T, n, d))
#         for t in range(T):
#             YA[t] = YA_list[t]

#     return YA


# def unfolded_n2v(As, d, p=1, q=1, sparse_matrix=False, flat=True, two_hop=False, num_walks=20, window=10, walklen=30, verbose=False, return_left=False):
#     """Computes the unfolded node2vec embedding"""

#     # TODO would like to be able to give rectangular matrices but this won't work in that case
#     # Assume fixed n over time
#     n = As[0].shape[0]
#     T = len(As)

#     # Construct the rectangular unfolded adjacency
#     if sparse_matrix:
#         A = As[0]
#         for t in range(1, T):
#             A = sparse.hstack((A, As[t]))
#     else:
#         if len(As.shape) == 2:
#             As = np.array([As[:, :]])
#         if len(As.shape) == 3:
#             T = len(As)
#             A = As[0, :, :]
#             for t in range(1, T):
#                 A = np.block([A, As[t]])
#     # T = len(As)
#     # A = As[0]
#     # for t in range(1, T):
#     #     A = sparse.hstack((A, As[t]))

#     # Construct the dilated unfolded adjacency matrix
#     DA = sparse.bmat([[None, A], [A.T, None]])
#     DA = sparse.csr_matrix(DA)

#     # Compute node2vec
#     n2v_obj = nodevectors.Node2Vec(
#         n_components=d,
#         epochs=num_walks,
#         walklen=walklen,
#         return_weight=1/p,
#         neighbor_weight=1/q,
#         verbose=verbose,
#         w2vparams={"window": window, "negative": 5, "iter": 10,
#                    "batch_words": 128}
#     )
#     if two_hop:
#         DA = DA @ DA.T

#     n2v = n2v_obj.fit_transform(DA)

#     # Take the rows of the embedding corresponding to the right embedding
#     # ([0:n] will be left embedding)
#     right_n2v = n2v[n:, :]
#     XA = n2v[0:n, :]

#     if flat:
#         YA = right_n2v
#     else:
#         YA = np.zeros((T, n, d))
#         for t in range(T):
#             YA[t] = right_n2v[t * n:(t + 1) * n, 0:d]

#     if not return_left:
#         return YA
#     else:
#         return XA, YA


# def regularised_unfolded_n2v(As, d, regulariser='auto', p=1, q=1, sparse_matrix=False, flat=True, two_hop=False, num_walks=20, window=10, walklen=30, verbose=False):
#     """Computes the unfolded node2vec embedding"""

#     # TODO would like to be able to give rectangular matrices but this won't work in that case
#     # Assume fixed n over time
#     n = As[0].shape[0]
#     T = len(As)

#     # Construct the rectangular unfolded adjacency
#     if sparse_matrix:
#         A = As[0]
#         for t in range(1, T):
#             A = sparse.hstack((A, As[t]))
#     else:
#         if len(As.shape) == 2:
#             As = np.array([As[:, :]])
#         if len(As.shape) == 3:
#             T = len(As)
#             A = As[0, :, :]
#             for t in range(1, T):
#                 A = np.block([A, As[t]])

#     Ls = to_laplacian(A, regulariser=regulariser)

#     # Construct the dilated unfolded adjacency matrix
#     DA = sparse.bmat([[None, Ls], [Ls.T, None]])
#     DA = sparse.csr_matrix(DA)

#     # Compute node2vec
#     n2v_obj = nodevectors.Node2Vec(
#         n_components=d,
#         epochs=num_walks,
#         walklen=walklen,
#         return_weight=1/p,
#         neighbor_weight=1/q,
#         verbose=verbose,
#         w2vparams={"window": window, "negative": 5, "iter": 10,
#                    "batch_words": 128}
#     )
#     if two_hop:
#         DA = DA @ DA.T

#     n2v = n2v_obj.fit_transform(DA)

#     # Take the rows of the embedding corresponding to the right embedding
#     # ([0:n] will be left embedding)
#     right_n2v = n2v[n:, :]

#     if flat:
#         YA = right_n2v
#     else:
#         YA = np.zeros((T, n, d))
#         for t in range(T):
#             YA[t] = right_n2v[t * n:(t + 1) * n, 0:d]

#     return YA


# def unfolded_prone(As, d, p=1, q=1, flat=True, two_hop=False, sparse_matrix=False):
#     """Computes the unfolded prone embedding"""

#     # Assume fixed n over time
#     n = As[0].shape[0]
#     T = len(As)

#     # Construct the rectangular unfolded adjacency
#     if sparse_matrix:
#         A = As[0]
#         for t in range(1, T):
#             A = sparse.hstack((A, As[t]))
#     else:
#         if len(As.shape) == 2:
#             As = np.array([As[:, :]])
#         if len(As.shape) == 3:
#             T = len(As)
#             A = As[0, :, :]
#             for t in range(1, T):
#                 A = np.block([A, As[t]])

#     # Construct the dilated unfolded adjacency matrix
#     DA = sparse.bmat([[None, A], [A.T, None]])
#     DA = sparse.csr_matrix(DA)

#     # Compute node2vec
#     n2v_obj = nodevectors.ProNE(
#         n_components=d,
#         verbose=False,
#     )
#     if two_hop:
#         DA = DA @ DA.T

#     n2v = n2v_obj.fit_transform(DA)

#     # Take the rows of the embedding corresponding to the right embedding
#     # ([0:n] will be left embedding)
#     right_n2v = n2v[n:, :]

#     if flat:
#         YA = right_n2v
#     else:
#         YA = np.zeros((T, n, d))
#         for t in range(T):
#             YA[t] = right_n2v[t * n:(t + 1) * n, 0:d]

#     return YA


def safe_inv_sqrt(a, tol=1e-12):
    """Computes the inverse square root, but returns zero if the result is either infinity
    or below a tolerance"""
    with np.errstate(divide="ignore"):
        b = 1 / np.sqrt(a)
    b[np.isinf(b)] = 0
    b[a < tol] = 0
    return b


def to_laplacian(A, regulariser=0):
    """Constructs the (regularised) symmetric Laplacian."""
    left_degrees = np.reshape(np.asarray(A.sum(axis=1)), (-1,))
    right_degrees = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
    if regulariser == "auto":
        regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
        print("Auto regulariser: {}".format(regulariser))
    left_degrees_inv_sqrt = safe_inv_sqrt(left_degrees + regulariser)
    right_degrees_inv_sqrt = safe_inv_sqrt(right_degrees + regulariser)
    L = sparse.diags(left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
    return L


def regularised_ULSE(
    As, d, regulariser=0, flat=True, sparse_matrix=False, return_left=False
):
    """Compute the unfolded (regularlised) Laplacian Spectral Embedding
    1. Construct dilated unfolded adjacency matrix
    2. Compute the symmetric (regularised) graph Laplacian
    3. Spectral Embed
    As: adjacency matrices of shape (T, n, n)
    d: embedding dimension
    regulariser: regularisation parameter. 0 for no regularisation, 'auto' for automatic regularisation (this often isn't the best).
    flat: True outputs embedding of shape (nT, d), False outputs shape (T, n, d)
    """

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

        # if len(As.shape) == 2:
        #     A = np.array([As[:, :]])
        # elif len(As.shape) == 3:
        #     T = len(As)
        #     A = As[0, :, :]
        #     for t in range(1, T):
        #         A = np.block([A, As[t]])
    # Construct (regularised) Laplacian matrix
    L = to_laplacian(A, regulariser=regulariser)

    # Compute spectral embedding
    U, S, Vt = sparse.linalg.svds(L, d)
    idx = np.abs(S).argsort()[::-1]
    YA_flat = Vt.T[:, idx] @ np.diag((np.sqrt(S[idx])))
    XA = U[:, idx] @ np.diag(np.sqrt(S[idx]))

    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t] = YA_flat[t * n : (t + 1) * n, 0:d]

    if not return_left:
        return YA
    else:
        return XA, YA


@nb.njit()
def form_omni_matrix(As, n, T):
    A = np.zeros((T * n, T * n))

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (As[t1] + As[t2]) / 2

    return A


def form_omni_matrix_sparse(As, n, T):
    A = sparse.lil_matrix((T * n, T * n))

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (As[t1] + As[t2]) / 2

    return A


def OMNI(As, d, flat=True, sparse_matrix=False):
    """Computes the omnibus spectral embedding"""

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct omnibus matrices
    if sparse_matrix:
        A = form_omni_matrix_sparse(As, n, T)
    else:
        A = form_omni_matrix(As, n, T)

    # Compute spectral embedding
    UA, SA, _ = sparse.linalg.svds(A, d)
    idx = SA.argsort()[::-1]
    UA = np.real(UA[:, idx][:, 0:d])
    SA = np.real(SA[idx][0:d])
    XA_flat = UA @ np.diag(np.sqrt(np.abs(SA)))

    if flat:
        XA = XA_flat
    else:
        XA = np.zeros((T, n, d))
        for t in range(T):
            XA[t] = XA_flat[t * n : (t + 1) * n, 0:d]

    return XA


def embed(
    As,
    d,
    method,
    regulariser=0,
    p=1,
    q=1,
    two_hop=False,
    verbose=False,
    num_walks=20,
    window=10,
    walklen=30,
):
    """Computes the embedding using a specified method"""
    if method.upper() == "SSE":
        YA = SSE(As, d)
    elif method.upper() == "SSE PROCRUSTES":
        YA = SSE(As, d, procrustes=True)
    elif method.upper() == "UASE":
        YA = UASE(As, d)
    elif method.upper() == "OMNI":
        YA = OMNI(As, d)
    elif method.upper() == "ULSE":
        YA = regularised_ULSE(As, d, regulariser=0)
    elif method.upper() == "URLSE":
        YA = regularised_ULSE(As, d, regulariser=regulariser)
    elif method.upper() == "INDEPENDENT NODE2VEC":
        YA = independent_n2v(
            As,
            d,
            p=p,
            q=q,
            num_walks=num_walks,
            window=window,
            walklen=walklen,
            verbose=verbose,
        )
    elif method.upper() == "UNFOLDED NODE2VEC":
        YA = unfolded_n2v(
            As,
            d,
            p=p,
            q=q,
            two_hop=two_hop,
            num_walks=num_walks,
            window=window,
            walklen=walklen,
            verbose=verbose,
        )
    elif method.upper() == "GLODYNE":
        YA = GloDyNE(As, d, num_walks=num_walks, window=window, walklen=walklen)
    else:
        raise Exception(
            "Method given is not a recognised embedding method\n- Please select from:\n\t> SSE\n\t> SSE PROCRUSTES\n\t> OMNI\n\t> UASE\n\t> ULSE\n\t> URLSE\n\t> INDEPENDENT NODE2VEC\n\t> UNFOLDED NODE2VEC\n\t> GLODYNE\n"
        )

    return YA


def ainv(x):
    return np.linalg.inv(x.T @ x) @ x.T


def online_embed(
    As,
    d,
    method,
    impact_times=[0],
    regulariser=None,
    p=1,
    q=1,
    sparse_matrix=False,
    two_hop=False,
    verbose=False,
    num_walks=20,
    window=10,
    walklen=30,
):
    """Computes the embedding using a specified method"""

    T = len(As)
    n = As[0].shape[0]
    impact_times = np.array(impact_times)

    Is = np.array([As[impact] for impact in impact_times])

    if method.upper() == "UASE":
        XA_partial, YA_partial = UASE(
            Is, d, return_left=True, sparse_matrix=sparse_matrix
        )
    elif method.upper() == "URLSE":
        XA_partial, YA_partial = regularised_ULSE(
            Is,
            d,
            regulariser=regulariser,
            return_left=True,
            sparse_matrix=sparse_matrix,
        )
    elif method.upper() == "UNFOLDED NODE2VEC":
        XA_partial, YA_partial = unfolded_n2v(
            Is,
            d,
            p=p,
            q=q,
            two_hop=two_hop,
            sparse_matrix=sparse_matrix,
            num_walks=num_walks,
            window=window,
            walklen=walklen,
            verbose=verbose,
            return_left=True,
        )
    else:
        raise Exception("Method given is not a recognised")

    YA_online = np.zeros((T, n, d))
    for t in range(T):
        if regulariser:
            YA_online[t] = (
                to_laplacian(As[t], regulariser=regulariser)
                @ XA_partial
                @ np.linalg.inv(XA_partial.T @ XA_partial)
            )
        else:
            YA_online[t] = As[t] @ XA_partial @ np.linalg.inv(XA_partial.T @ XA_partial)

    return YA_online


def plot_embedding(ya, n, T, tau, return_df=False, title=None):
    if len(ya.shape) == 3:
        ya = ya.reshape((n * T, -1))

    yadf = pd.DataFrame(ya[:, 0:2])
    yadf.columns = ["Dimension {}".format(i + 1) for i in range(yadf.shape[1])]
    yadf["Time"] = np.repeat([t for t in range(T)], n)
    yadf["Community"] = list(tau) * T
    yadf["Community"] = yadf["Community"].astype(str)
    pad_x = (max(ya[:, 0]) - min(ya[:, 0])) / 50
    pad_y = (max(ya[:, 1]) - min(ya[:, 1])) / 50
    fig = px.scatter(
        yadf,
        x="Dimension 1",
        y="Dimension 2",
        color="Community",
        animation_frame="Time",
        range_x=[min(ya[:, 0]) - pad_x, max(ya[:, 0]) + pad_x],
        range_y=[min(ya[:, 1]) - pad_y, max(ya[:, 1]) + pad_y],
    )
    if title:
        fig.update_layout(title=title)

    fig.show()
    if return_df:
        return yadf


def test_mean_change(ya1, ya2, tau_permutation=None, n_sim=1000):
    """Mean change permutation test from two embedding sets.

    Use this function for spatial testing. If you require temporal testing - see vector displacement test.

    Null hypothesis: The two embedding sets come from the same distribution.
    Alternative: They two embedding sets come from different distributions

    ya1 & ya2: (numpy arrays) Two equal-shape embedding sets
    tau_permutation: (numpy array) Known community allocations - no need to use in spatial testing case.
    n_sim: (int) Number of permuted test statistics computed.
    """
    print("USING MEAN CHANGE")

    # Compute mean difference between observed sets
    obs1 = np.mean(ya1, axis=0)
    obs2 = np.mean(ya2, axis=0)
    t_obs = np.linalg.norm(obs1 - obs2)

    n_1 = ya1.shape[0]
    n_2 = ya2.shape[0]

    # Compute permuted test statistics
    if tau_permutation is not None:

        # Get the idx of the permutable groups (in this case groups of tau=0 or tau=1)
        ya_all = np.row_stack([ya1, ya2])
        idx_group_1 = np.where(np.tile(tau_permutation, 2) == 0)[0]
        idx_group_2 = np.where(np.tile(tau_permutation, 2) == 1)[0]

        # Permute the idx groups
        t_obs_stars = np.zeros((n_sim,))
        for i in range(n_sim):
            idx_group_1_shuffled = idx_group_1.copy()
            idx_group_2_shuffled = idx_group_2.copy()

            np.random.shuffle(idx_group_1_shuffled)
            np.random.shuffle(idx_group_2_shuffled)

            ya_all_perm = np.zeros((ya1.shape[0] * 2, ya1.shape[1]))
            ya_all_perm[idx_group_1_shuffled] = ya_all[idx_group_1]
            ya_all_perm[idx_group_2_shuffled] = ya_all[idx_group_2]

            # Split back into the two time sets, now including the temporal permutations
            ya_star_1 = ya_all_perm[0:n_1, :]
            ya_star_2 = ya_all_perm[n_2:, :]

            # Compute the permuted test statistic
            obs_star_1 = np.mean(ya_star_1, axis=0)
            obs_star_2 = np.mean(ya_star_2, axis=0)

            t_obs_star = np.linalg.norm(obs_star_1 - obs_star_2)
            # t_obs_star = test_statistic(ya_star_1, ya_star_2)
            t_obs_stars[i] = t_obs_star

    else:
        # Permute all nodes
        ya_all = np.row_stack([ya1, ya2])
        n_1 = ya1.shape[0]
        n_2 = ya2.shape[0]
        t_obs_stars = np.zeros((n_sim,))
        for i in range(n_sim):
            # Create two randomly sets from the observed sets
            idx_shuffled = np.arange(n_1 + n_2)
            np.random.shuffle(idx_shuffled)
            idx_1 = idx_shuffled[0:n_1]
            idx_2 = idx_shuffled[n_1 : n_1 + n_2]
            ya_star_1 = ya_all[idx_1]
            ya_star_2 = ya_all[idx_2]

            # Calculate test statistic for permuted vectors
            obs_star_1 = np.mean(ya_star_1, axis=0)
            obs_star_2 = np.mean(ya_star_2, axis=0)
            t_obs_star = np.linalg.norm(obs_star_1 - obs_star_2)

            # Add permuted test statistic to null distribution
            t_obs_stars[i] = t_obs_star

    # p-value is given by how extreme the observed test statistic is vs the null distribution
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)

    return p_hat


@nb.njit()
def vector_displacement_test(ya1, ya2):
    """Computes the vector displacement between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)

    # Magnitude of average displacement vector
    t_obs = np.linalg.norm(sum_axis)

    return t_obs


@nb.njit()
def masked_vector_displacement_test(ya1, ya2, mask=None):
    """Computes the vector displacement between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    displacement = ya2 - ya1
    if mask is not None:
        displacement = displacement[mask]

    sum_axis = displacement.sum(axis=0)

    # Magnitude of average displacement vector
    t_obs = np.linalg.norm(sum_axis)

    return t_obs


@nb.njit
def faster_inner_product_distance(ya1, ya2):
    """Computes the inner product distance between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    inner_product_dists = 1 - np.sum(ya1 * ya2, axis=1)
    sum_axis = inner_product_dists.sum(axis=0)

    # Magnitude of average displacement vector
    # t_obs = np.linalg.norm(sum_axis)

    return sum_axis


@nb.njit
def cosine_distance(ya1, ya2):
    norms = np.linalg.norm(ya1, axis=1) * np.linalg.norm(ya2, axis=1)
    mask = np.where(norms != 0)
    dot_product = np.dot(ya1[mask], ya2[mask].T)
    distances = 1 - (dot_product / norms[mask])
    sum_axis = distances.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)
    return t_obs


@nb.njit()
def test_temporal_displacement(ya, n, T, changepoint, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    This function should only be used when comparing more than two time points across a single changepoint.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    T: (int) number of time points
    radius: (int) number of time points to permute before/after the changepoint
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """

    if changepoint < 1:
        raise Exception("Changepoint must be at least 1.")
    elif changepoint > T:
        raise Exception("Changepoint must be less than or equal T")

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * changepoint : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = vector_displacement_test(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        # for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            possible_perms = ya[j::n, :]
            ya_star[j::n, :] = possible_perms[np.random.choice(T, T, replace=False), :]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * changepoint, :]
        ya_star_2 = ya_star[n * changepoint : n * (changepoint + 1), :]
        t_obs_star = vector_displacement_test(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


@nb.njit()
def test_temporal_displacement_two_times(ya, n, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    This can be used in the case where you are comparing ONLY two time points.
    In this case it's much faster than the above function.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """
    # Number of time points must = 2 for this method
    T = 2

    # Select time point embedding just before and after the changepoint
    ya1 = ya[0:n, :]
    ya2 = ya[n : 2 * n, :]

    # Get observed value of the test
    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)

    # Permute the sets
    t_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):

        # Randomly swap the signs of each row of displacement
        # signs = np.random.choice([-1, 1], n)
        signs = np.random.randint(0, 2, size=displacement.shape) * 2 - 1
        displacement_permuted = displacement * signs
        sum_axis_permuted = displacement_permuted.sum(axis=0)
        t_star = np.linalg.norm(sum_axis_permuted)
        t_stars[sim_iter] = t_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_stars >= t_obs)
    return p_hat


def test_temporal_displacement_two_times_not_jit(ya, n, radius=1, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    radius: (int) number of time points to permute before/after the changepoint
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """
    # Number of time points must = 2 for this method
    T = 2

    # Select time point embedding just before and after the changepoint
    ya1 = ya[0:n, :]
    ya2 = ya[n : 2 * n, :]

    # Get observed value of the test
    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)

    # Permute the sets
    t_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):

        # Randomly swap the signs of each row of displacement
        # signs = np.random.choice([-1, 1], n)
        signs = np.random.randint(0, 2, size=displacement.shape) * 2 - 1
        displacement_permuted = displacement * signs
        sum_axis_permuted = displacement_permuted.sum(axis=0)
        t_star = np.linalg.norm(sum_axis_permuted)
        t_stars[sim_iter] = t_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_stars >= t_obs)
    return p_hat


@nb.njit()
def mean_change_test_stat(ya1, ya2):
    ya1_mean_cols = np.zeros((ya1.shape[1]))
    ya2_mean_cols = np.zeros((ya1.shape[1]))
    for colidx in range(ya1.shape[1]):
        ya1_mean_cols[colidx] = np.mean(ya1[:, colidx])
        ya2_mean_cols[colidx] = np.mean(ya2[:, colidx])

    ya_mean_col_diff = ya1_mean_cols - ya2_mean_cols
    t_obs = np.linalg.norm(ya_mean_col_diff)
    return t_obs


@nb.njit()
def test_graph_mean_change(ya, n, T, changepoint, n_sim=1000, perm_range=2):
    """
    Computes a mean change permutation test with temporal node permutations.

    changepoint: (int 1-T) the first time point of a change.
    perm_range: (int > 1) number of time points either side of the changepoint from which permutations can be taken.
        If changepoint = 5, then permuations will be taken from times 3,4,5,6.
    """

    # If the perm_range was 1, only 0 p-values would be returned as the test statistic will be the same, no matter the permutation
    if perm_range <= 1:
        raise Exception("The permutation range must be at least 2.")

    if changepoint + perm_range > T:
        raise Exception(
            "The permuation range goes above the number of available time points."
        )

    T_for_perms = perm_range * 2

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * (changepoint) : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = mean_change_test_stat(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            # Restrict the possible permutations to perm_range around the changepoint
            possible_perms = ya[
                np.array(
                    [
                        j + t * n
                        for t in range(
                            changepoint - perm_range, changepoint + perm_range
                        )
                    ]
                ),
                :,
            ]

            # times_for_perms = np.arange(
            #     changepoint - perm_range, changepoint + perm_range)
            # possible_perms = possible_perms[times_for_perms, :]

            ya_star[
                np.array(
                    [
                        j + t * n
                        for t in range(
                            changepoint - perm_range, changepoint + perm_range
                        )
                    ]
                ),
                :,
            ] = possible_perms[
                np.random.choice(T_for_perms, T_for_perms, replace=False), :
            ]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * (changepoint), :]
        ya_star_2 = ya_star[n * (changepoint) : n * (changepoint + 1), :]

        t_obs_star = mean_change_test_stat(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


@nb.njit()
def test_graph_mean_change_asym(
    ya, n, T, changepoint, n_sim=1000, perm_range=1, before=True
):
    """
    Computes a mean change permutation test with temporal node permutations.

    changepoint: (int 1-T) the first time point of a change.
    perm_range: (int > 0) number of time points either side of the changepoint from which permutations can be taken.
    before: (bool) should the extra non-testing time point be before[True] or after[False] the change?

        If changepoint=5 and before=True, then permuations will be taken from times 3,4,5.
        If changepoint=5 and before=False, then permuations will be taken from times 4,5,6.
    """

    # If the perm_range was 1, only 0 p-values would be returned as the test statistic will be the same, no matter the permutation
    if perm_range < 1:
        raise Exception("The permutation range must be at least 1.")

    if changepoint + perm_range > T:
        raise Exception(
            "The permuation range goes above the number of available time points."
        )

    T_for_perms = perm_range * 2 + 1

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * (changepoint) : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = mean_change_test_stat(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            # Restrict the possible permutations to perm_range around the changepoint
            if before:
                # Allow the nodes to permute more times before the change
                possible_perms = ya[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - (perm_range + 1), changepoint + perm_range
                            )
                        ]
                    ),
                    :,
                ]

                ya_star[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - (perm_range + 1), changepoint + perm_range
                            )
                        ]
                    ),
                    :,
                ] = possible_perms[
                    np.random.choice(T_for_perms, T_for_perms, replace=False), :
                ]

            else:
                # Allow the nodes to permute more times after the change
                possible_perms = ya[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - perm_range, changepoint + (perm_range + 1)
                            )
                        ]
                    ),
                    :,
                ]

                ya_star[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - perm_range, changepoint + (perm_range + 1)
                            )
                        ]
                    ),
                    :,
                ] = possible_perms[
                    np.random.choice(T_for_perms, T_for_perms, replace=False), :
                ]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * (changepoint), :]
        ya_star_2 = ya_star[n * (changepoint) : n * (changepoint + 1), :]

        t_obs_star = mean_change_test_stat(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


@nb.njit()
def test_graph_mean_change_no_perm_rest(ya, n, T, changepoint, n_sim=1000):
    """
    Same as above but with no restriction on time points to permute from

    Computes a mean change permutation test with temporal node permutations.

    changepoint: (int 1-T) the first time point of a change.
    perm_range: (int > 1) number of time points either side of the changepoint from which permutations can be taken.
        If changepoint = 5, then permuations will be taken from times 3,4,5,6.
    """

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * (changepoint) : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = mean_change_test_stat(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            # For each node get its position at each time - permute over these positions
            possible_perms = ya[np.array([j + t * n for t in range(T)]), :]
            ya_star[np.array([j + t * n for t in range(T)]), :] = possible_perms[
                np.random.choice(T, T, replace=False), :
            ]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * (changepoint), :]
        ya_star_2 = ya_star[n * (changepoint) : n * (changepoint + 1), :]

        t_obs_star = mean_change_test_stat(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


def unrealistic_ideal_testing(
    ya1, ya2, return_p=True, plot=False, title="", n_sim=1000
):

    def test_statistic(ya1, ya2):
        obs1 = np.mean(ya1, axis=0)
        obs2 = np.mean(ya2, axis=0)

        t_obs = np.linalg.norm(obs1 - obs2)
        return t_obs

    t_obs = test_statistic(ya1, ya2)

    n_1 = ya1.shape[0]
    n_2 = ya2.shape[0]

    ya_all = np.row_stack([ya1, ya2])

    # Permute all nodes
    n_1 = ya1.shape[0]
    n_2 = ya2.shape[0]
    t_obs_stars = np.zeros((n_sim,))
    for i in range(n_sim):
        idx_shuffled = np.arange(n_1 + n_2)
        np.random.shuffle(idx_shuffled)
        idx_1 = idx_shuffled[0:n_1]
        idx_2 = idx_shuffled[n_1 : n_1 + n_2]
        ya_star_1 = ya_all[idx_1]
        ya_star_2 = ya_all[idx_2]

        # Calculate test statistic for permuted vectors
        t_obs_star = test_statistic(ya_star_1, ya_star_2)
        t_obs_stars[i] = t_obs_star

    p_hat = 1 / n_sim * sum(t_obs_stars >= t_obs)

    if plot:
        sns.displot(x=t_obs_stars)
        plt.axvline(t_obs, 0, c="red")
        plt.title(title + r": $\hat{p} = $" + "%.3f" % (p_hat))
        # plt.title(r"title: $\hat{p} = {}$".format(p_hat))
        plt.show()
    if return_p:
        return p_hat


def dim_select(As, d, plot=True, sparse_matrix=False):
    """This is Ian's function that I stole"""
    # Construct rectangular matrices
    T = len(As)
    A = As[0]
    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        if len(As.shape) == 2:
            As = np.array([As[:, :]])
        if len(As.shape) == 3:
            T = len(As)
            A = As[0, :, :]
            for t in range(1, T):
                A = np.block([A, As[t]])

    UA, SA, VAt = sparse.linalg.svds(A, d)
    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    SA = SA[idx]

    # Compute likelihood profile
    n = len(SA)
    lq = np.zeros(n)
    lq[0] = "nan"
    for q in range(1, n):
        theta_0 = np.mean(SA[:q])
        theta_1 = np.mean(SA[q:])
        sigma = np.sqrt(
            ((q - 1) * np.var(SA[:q]) + (n - q - 1) * np.var(SA[q:])) / (n - 2)
        )
        lq_0 = np.sum(np.log(stats.norm.pdf(SA[:q], theta_0, sigma)))
        lq_1 = np.sum(np.log(stats.norm.pdf(SA[q:], theta_1, sigma)))
        lq[q] = lq_0 + lq_1
    lq_best = np.nanargmax(lq)

    if plot:
        plotrange = d
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.0))
        plt.subplots_adjust(hspace=0.3)

        axs[0].plot(range(plotrange), SA[:plotrange], ".-")
        axs[0].set_title("Singular values")
        axs[0].set_xlabel("Number of dimensions")
        axs[0].axvline(x=lq_best, ls="--", c="k")

        axs[1].plot(range(plotrange), lq[:plotrange], ".-")
        axs[1].set_title("Log likelihood")
        axs[1].set_xlabel("Number of dimensions")
        axs[1].axvline(x=lq_best, ls="--", c="k")

    return lq_best


def spherical_degree_correct(YA, T):
    """Degree corrects by plotting the ray angles

    (This is probably the worst, most inefficient piece of code I've ever written)
    """

    if len(YA.shape) == 2:
        flat = True
        n = int(YA.shape[0] / T)
        d = YA.shape[1]
    elif len(YA.shape) == 3:
        flat = False
        n = YA[0].shape[0]
        d = YA[0].shape[1]

    def spherical_embed(x):
        d = len(x)
        theta = np.zeros(d - 1)

        if x[0] > 0:
            theta[0] = np.arccos(x[1] / np.linalg.norm(x[:2]))
        else:
            theta[0] = 2 * np.pi - np.arccos(x[1] / np.linalg.norm(x[:2]))

        for i in range(d - 1):
            theta[i] = np.arccos(x[i + 1] / np.linalg.norm(x[: (i + 2)]))

        return theta

    if flat:
        YD = np.zeros((n * T, d - 1))
        for t in range(T):
            for i in range(n):
                if np.linalg.norm(YA[n * t + i, :]) > 1e-10:
                    YD[n * t + i, :] = spherical_embed(YA[n * t + i, :])
    else:
        YD = np.zeros((T, n, d - 1))
        for t in range(T):
            for i in range(n):
                if np.linalg.norm(YA[t, i]) > 1e-10:
                    YD[t, i] = spherical_embed(YA[t, i])

    return YD


# @nb.njit
def general_unfolded_matrix(As, sparse_matrix=False):
    """Forms the general unfolded matrix from an adjacency series"""
    T = len(As)
    n = As[0].shape[0]

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))

        # Construct the dilated unfolded adjacency matrix
        DA = sparse.bmat([[None, A], [A.T, None]])
        DA = sparse.csr_matrix(DA)
    else:
        A = As[0]
        for t in range(1, T):
            A = np.block([A, As[t]])

        DA = np.zeros((n + n * T, n + n * T))
        DA[0:n, n:] = A
        DA[n:, 0:n] = A.T

    return DA
