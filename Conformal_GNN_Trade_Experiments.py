# %%
import copy
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.linalg import block_diag
from tqdm import tqdm

import torch
from torch.functional import F

import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv, GCNConv

from scipy import sparse

# Specific to trade
from decoder import NodePredictor
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed

from torch_geometric.utils import to_scipy_sparse_matrix

# %% [markdown]
# ## Experiment parameters

# %%
# print("\n\n\nWARNING: NOT SAVING\n\n\n")

dataset_name = "Trade"


# Training/validation/calibration/test dataset split sizes
props = np.array([0.2, 0.1, 0.35, 0.35])

# Target 1-coverage for conformal prediction
alpha = 0.1

# # # Number of experiments
# print("REduced params!\n\n")
# num_train_trans = 1
# num_permute_trans = 1
# num_train_semi_ind = 1

# # GNN model parameters
# print("REduced GNN params!\n\n")
# num_epochs = 1
# num_channels_GCN = 32
# num_channels_GAT = 32

# Number of experiments
num_train_trans = 10
num_permute_trans = 100
num_train_semi_ind = 50

# GNN model parameters
num_epochs = 40
num_channels_GCN = 32
num_channels_GAT = 32


# Save results
results_file = f"results/Conformal_GNN_{dataset_name}_Results.pkl"

# %% [markdown]
# ## Load dataset

# %%
edges_df = pd.read_csv("datasets/tgbn-trade/tgbn-trade_edgelist.csv")

nation = edges_df["nation"].values
trading_nation = edges_df["trading nation"].values
unique_countries = np.unique(np.concatenate((nation, trading_nation)))
n = len(unique_countries)

country_to_idx = {country: idx for idx, country in enumerate(unique_countries)}
idx_to_country = {idx: country for country, idx in country_to_idx.items()}

srcs = [country_to_idx[country] for country in nation]
dsts = [country_to_idx[country] for country in trading_nation]
weights = edges_df["weight"].values
times = np.unique(edges_df["year"].values)

As = []
for t in tqdm(times):
    if t == 2016:
        continue

    mask = edges_df["year"] == t
    srcs_t = torch.tensor(srcs)[mask]
    dsts_t = torch.tensor(dsts)[mask]
    edge_index = torch.stack([srcs_t, dsts_t])
    edge_weight = torch.tensor(weights)[mask]

    A = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=n)
    As.append(A)

T = len(As)

# %%
labels_df = pd.read_csv("datasets/tgbn-trade/tgbn-trade_node_labels.csv")

# For each nation at each year, find the trading nation with the highest weight
# This will be the label for the node

data_mask = np.array([[True] * T for _ in range(n)])
labels = np.zeros((n, T))
# 1: to skip 1986 as we don't have a label for that
for year_idx, year in tqdm(enumerate(times[1:])):
    label_df_t = labels_df[labels_df["year"] == year]

    for country in unique_countries:
        country_enc = country_to_idx[country]

        label_df_country = label_df_t[label_df_t["nation"] == country]

        if len(label_df_country) == 0:
            labels[country_enc, year_idx] = np.nan
            data_mask[country_enc, year_idx] = False
            continue

        max_weight = label_df_country["weight"].max()
        max_weight_country = label_df_country[label_df_country["weight"] == max_weight][
            "trading nation"
        ].values[0]

        labels[country_enc, year_idx] = country_to_idx[max_weight_country]

for t in range(T):
    data_mask[np.where(np.sum(As[t], axis=0) == 0)[0], t] = False

flat_labels = labels.flatten()
node_labels_enc = pd.Categorical(flat_labels)
node_labels = node_labels_enc.codes

num_classes = len(node_labels_enc.categories)

assert node_labels.shape[0] == n * T
# %%
masked_labels = node_labels[data_mask.flatten()]
time_labels = np.repeat(np.arange(T), n)
masked_time_labels = time_labels[data_mask.flatten()]

len(masked_time_labels)

avg_labels = []
for time in np.unique(masked_time_labels):
    time_mask = masked_time_labels == time
    avg_labels.append(np.mean(masked_labels[time_mask]))

# %%
num_edges = []
for A in As:
    A_binary = np.where(A.todense() > 0, 1, 0)
    num_edges.append(np.sum(A_binary))


# %%
edges = 0
for t in range(T):
    A_t = As[t].todense()
    A_binary = np.where(A_t > 0, 1, 0)
    edges += np.sum(A_binary)


# %%
class Dynamic_Network(Dataset):
    """
    A pytorch geometric dataset for a dynamic network.
    """

    def __init__(self, As, labels):
        self.As = As
        self.T = len(As)
        self.n = As[0].shape[0]
        self.classes = labels

        assert len(labels) == self.n * self.T

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):
        x = torch.sparse.spdiags(
            torch.ones(self.n),
            offsets=torch.tensor([0]),
            shape=(self.n, self.n),
        )
        edge_index = torch.tensor(
            np.array([self.As[idx].nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = torch.tensor(np.array(self.As[idx].data), dtype=torch.float)
        y = torch.tensor(
            self.classes[self.n * idx : self.n * (idx + 1)], dtype=torch.long
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight, y=y
        )
        data.num_nodes = self.n

        return data


# %%
class Block_Diagonal_Network(Dataset):
    """
    A pytorch geometric dataset for the block diagonal version of a Dynamic Network object.
    """

    def __init__(self, dataset):
        self.A = sparse.block_diag(dataset.As)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

        assert len(self.classes) == self.n * self.T

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.sparse.spdiags(
            torch.ones(self.n * (self.T)),
            offsets=torch.tensor([0]),
            shape=(self.n * (self.T), self.n * (self.T)),
        )
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = torch.tensor(np.array(self.A.data), dtype=torch.float)
        y = torch.tensor(self.classes, dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight, y=y
        )
        data.num_nodes = self.n * self.T

        return data


# %%
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


class Unfolded_Network(Dataset):
    """
    A pytorch geometric dataset for the dilated unfolding of a Dynamic Network object.
    """

    def __init__(self, dataset):
        self.A = general_unfolded_matrix(dataset.As, sparse_matrix=True)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

        assert len(self.classes) == self.n * self.T

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.sparse.spdiags(
            torch.ones(self.n * (self.T + 1)),
            offsets=torch.tensor([0]),
            shape=(self.n * (self.T + 1), self.n * (self.T + 1)),
        )
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = torch.tensor(np.array(self.A.data), dtype=torch.float)

        # Add n zeros to the start of y for the anchors
        y = torch.tensor(
            np.concatenate((np.zeros(self.n), self.classes)), dtype=torch.long
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight, y=y
        )
        data.num_nodes = self.n * (self.T + 1)

        return data


# %%
dataset = Dynamic_Network(As, node_labels)
dataset_BD = Block_Diagonal_Network(dataset)[0]
dataset_UA = Unfolded_Network(dataset)[0]

# %% [markdown]
# Define general GCN and GAT neural networks to be applied to both the block diagonal and dilated unfolded networks.


# %%
class GCN(torch.nn.Module):
    def __init__(self, num_nodes, num_channels, num_classes, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(num_nodes, num_channels)
        self.conv2 = GCNConv(num_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class GAT(torch.nn.Module):
    def __init__(self, num_nodes, num_channels, num_classes, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GATConv(num_nodes, num_channels)
        self.conv2 = GATConv(num_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x


# %%
def train(model, data, train_mask):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()

    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    return loss


def valid(model, data, valid_mask):
    model.eval()

    out = model(data.x, data.edge_index, data.edge_weight)
    pred = out.argmax(dim=1)
    correct = pred[valid_mask] == data.y[valid_mask]
    acc = int(correct.sum()) / int(valid_mask.sum())

    return acc


# %% [markdown]
# ## Data split functions

# %% [markdown]
# Create a data mask that consists only of useful nodes for training and testing. We only consider nodes in a particular time window if it has degree greater than zero at that time window. Also, we ignore nodes with label `Teacher` meaning that the number of classes is now 10.

# %%
# data_mask = np.array([[True] * T for _ in range(n)])

# for t in range(T):
#     data_mask[np.where(np.sum(As[t], axis=0) == 0)[0], t] = False
#     data_mask[np.where(node_labels == label_dict['Teachers'])[0], t] = False

# num_classes = np.unique(node_labels).shape[0]
print(
    f"Percentage of usable node/time pairs: {100 * np.sum(data_mask) / (n * T) :02.1f}%"
)


# %%


def mask_split(mask, split_props, seed=0, mode="transductive"):
    np.random.seed(seed)

    n, T = mask.shape

    if mode == "transductive":
        # Flatten mask array into one dimension in blocks of nodes per time
        flat_mask = mask.reshape(-1)
        n_masks = np.sum(flat_mask)

        # Split shuffled flatten mask array indices into correct proportions
        flat_mask_idx = np.where(flat_mask)[0]
        np.random.shuffle(flat_mask_idx)
        split_ns = np.cumsum([round(n_masks * prop) for prop in split_props[:-1]])
        split_idx = np.split(flat_mask_idx, split_ns)

    if mode == "semi-inductive":
        flat_mask = mask.reshape(-1)

        # Find time such that final proportion of masks happen after that time
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        flat_mask_idx = np.where(flat_mask)[0]
        flat_mask_start_idx = flat_mask_idx[: n * T_trunc]
        flat_mask_end_idx = flat_mask_idx[n * T_trunc :]

        n_masks_start = len(flat_mask_start_idx)

        # Split starting shuffled flatten mask array into correct proportions
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-1] / np.sum(split_props[:-1])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)

        # Place finishing flatten mask array at the end
        split_idx.append(flat_mask_end_idx)

    split_masks = np.array([[False] * n * T for _ in range(len(split_props))])
    for i in range(len(split_props)):
        split_masks[i, split_idx[i]] = True

    return split_masks


# %%
def mask_mix(mask_1, mask_2, seed=0):
    np.random.seed(seed)

    n = len(mask_1)
    n1 = np.sum(mask_1)
    n2 = np.sum(mask_2)

    mask_idx = np.where(mask_1 + mask_2)[0]
    np.random.shuffle(mask_idx)
    split_idx = np.split(mask_idx, [n1])

    split_masks = np.array([[False] * n for _ in range(2)])
    for i in range(2):
        split_masks[i, split_idx[i]] = True

    return split_masks


# %% [markdown]
# Testing the training/validation/calibration/test data split functions.

# %%
props = np.array([0.2, 0.1, 0.35, 0.35])
train_mask, valid_mask, calib_mask, test_mask = mask_split(
    data_mask, props, mode="semi-inductive"
)

print(f"Percentage train: {100 * np.sum(train_mask) / np.sum(data_mask) :02.1f}%")
print(f"Percentage valid: {100 * np.sum(valid_mask) / np.sum(data_mask) :02.1f}%")
print(f"Percentage calib: {100 * np.sum(calib_mask) / np.sum(data_mask) :02.1f}%")
print(f"Percentage test:  {100 * np.sum(test_mask)  / np.sum(data_mask) :02.1f}%")

# %% [markdown]
# ## Conformal prediction functions


# %%
def get_prediction_sets(output, data, calib_mask, test_mask, alpha=0.1):
    n_calib = calib_mask.sum()
    # output = model(data.x, data.edge_index, data.edge_weight)

    # Compute softmax probabilities
    smx = torch.nn.Softmax(dim=1)
    calib_heuristic = smx(output[calib_mask]).detach().numpy()
    test_heuristic = smx(output[test_mask]).detach().numpy()

    # APS
    calib_pi = calib_heuristic.argsort(1)[:, ::-1]
    calib_srt = np.take_along_axis(calib_heuristic, calib_pi, axis=1).cumsum(axis=1)
    calib_scores = np.take_along_axis(calib_srt, calib_pi.argsort(axis=1), axis=1)[
        range(n_calib), data.y[calib_mask]
    ]

    # Get the score quantile
    qhat = np.quantile(
        calib_scores, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib, method="higher"
    )

    test_pi = test_heuristic.argsort(1)[:, ::-1]
    test_srt = np.take_along_axis(test_heuristic, test_pi, axis=1).cumsum(axis=1)
    pred_sets = np.take_along_axis(test_srt <= qhat, test_pi.argsort(axis=1), axis=1)

    return pred_sets


# %%
def accuracy(output, data, test_mask):
    # output = model(data.x, data.edge_index, data.edge_weight)
    pred = output.argmax(dim=1)
    correct = pred[test_mask] == data.y[test_mask]
    acc = int(correct.sum()) / int(test_mask.sum())

    return acc


# %%
def avg_set_size(pred_sets, test_mask):
    return np.mean(np.sum(pred_sets, axis=1))


# %%
def coverage(pred_sets, data, test_mask):
    in_set = np.array(
        [pred_set[label] for pred_set, label in zip(pred_sets, data.y[test_mask])]
    )

    return np.mean(in_set)


# %% [markdown]
# ## GNN training

# %% [markdown]
# Initiate nested results data structure.

# %%
results = {}

methods = ["BD", "UA"]
GNN_models = ["GCN", "GAT"]
regimes = ["Trans", "Semi-Ind"]
outputs = ["Accuracy", "Avg Size", "Coverage"]
times = ["All"] + list(range(T))

for method in methods:
    results[method] = {}

    for GNN_model in GNN_models:
        results[method][GNN_model] = {}

        for regime in regimes:
            results[method][GNN_model][regime] = {}

            for output in outputs:
                results[method][GNN_model][regime][output] = {}

                for time in times:
                    results[method][GNN_model][regime][output][time] = []

# %% [markdown]
# ### Transductive experiments

# %%
# %%time

for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_trans):
        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, mode="transductive"
        )

        if method == "BD":
            method_str = "Block Diagonal"
            data = dataset_BD
        if method == "UA":
            method_str = "Unfolded"
            data = dataset_UA
            # Pad masks to include anchor nodes
            train_mask = np.concatenate((np.array([False] * n), train_mask))
            valid_mask = np.concatenate((np.array([False] * n), valid_mask))
            calib_mask = np.concatenate((np.array([False] * n), calib_mask))
            test_mask = np.concatenate((np.array([False] * n), test_mask))

        if GNN_model == "GCN":
            model = GCN(data.num_nodes, num_channels_GCN, num_classes, seed=i)
        if GNN_model == "GAT":
            model = GAT(data.num_nodes, num_channels_GAT, num_classes, seed=i)

        optimizer = torch.optim.Adam(model.parameters())

        print(f"Training {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask)
            valid_acc = valid(model, data, valid_mask)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        best_output = best_model(data.x, data.edge_index, data.edge_weight)
        print(f"Best valid: {max_valid_acc:.4f}")
        print(f"Evaluating {method_str} {GNN_model} Number {i}")

        coverage_list = []
        for j in tqdm(range(num_permute_trans)):
            # Permute the calibration and test datasets
            calib_mask, test_mask = mask_mix(calib_mask, test_mask, seed=j)

            pred_sets = get_prediction_sets(
                best_output, data, calib_mask, test_mask, alpha
            )

            cov = coverage(pred_sets, data, test_mask)
            coverage_list.append(cov)
            results[method][GNN_model]["Trans"]["Accuracy"]["All"].append(
                accuracy(best_output, data, test_mask)
            )
            results[method][GNN_model]["Trans"]["Avg Size"]["All"].append(
                avg_set_size(pred_sets, test_mask)
            )
            results[method][GNN_model]["Trans"]["Coverage"]["All"].append(cov)

            for t in range(T):
                # Consider test nodes only at time t
                if method == "BD":
                    time_mask = np.array([[False] * n for _ in range(T)])
                    time_mask[t] = True
                    time_mask = time_mask.reshape(-1)
                if method == "UA":
                    time_mask = np.array([[False] * n for _ in range(T + 1)])
                    time_mask[t + 1] = True
                    time_mask = time_mask.reshape(-1)

                test_mask_t = time_mask * test_mask
                if np.sum(test_mask_t) == 0:
                    continue

                # Get prediction sets corresponding to time t
                index_mapping = {
                    index: i for i, index in enumerate(np.where(test_mask)[0])
                }
                indices = [index_mapping[index] for index in np.where(test_mask_t)[0]]
                pred_sets_t = pred_sets[np.array(indices)]

                results[method][GNN_model]["Trans"]["Accuracy"][t].append(
                    accuracy(best_output, data, test_mask_t)
                )
                results[method][GNN_model]["Trans"]["Avg Size"][t].append(
                    avg_set_size(pred_sets_t, test_mask_t)
                )
                results[method][GNN_model]["Trans"]["Coverage"][t].append(
                    coverage(pred_sets_t, data, test_mask_t)
                )

        print("Coverage: ", np.mean(coverage_list))

# %% [markdown]
# ### Semi-inductive experiments

# %%
# %%time

for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_semi_ind):
        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, mode="semi-inductive"
        )

        if method == "BD":
            method_str = "Block Diagonal"
            data = dataset_BD
        if method == "UA":
            method_str = "Unfolded"
            data = dataset_UA
            # Pad masks to include anchor nodes
            train_mask = np.concatenate((np.array([False] * n), train_mask))
            valid_mask = np.concatenate((np.array([False] * n), valid_mask))
            calib_mask = np.concatenate((np.array([False] * n), calib_mask))
            test_mask = np.concatenate((np.array([False] * n), test_mask))

        if GNN_model == "GCN":
            model = GCN(data.num_nodes, num_channels_GCN, num_classes, seed=i)
        if GNN_model == "GAT":
            model = GAT(data.num_nodes, num_channels_GAT, num_classes, seed=i)

        optimizer = torch.optim.Adam(model.parameters())

        print(f"Training {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        val_curve = []
        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask)
            valid_acc = valid(model, data, valid_mask)
            val_curve.append(valid_acc)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        best_output = best_model(data.x, data.edge_index, data.edge_weight)
        print(f"Best valid: {max_valid_acc:.4f}")
        print(f"Evaluating {method_str} {GNN_model} Number {i}")

        # Cannot permute the calibration and test datasets in semi-inductive experiments

        pred_sets = get_prediction_sets(best_output, data, calib_mask, test_mask, alpha)

        cov = coverage(pred_sets, data, test_mask)
        print(f"Coverage: {cov:.4f}")
        results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"].append(
            accuracy(best_output, data, test_mask)
        )
        results[method][GNN_model]["Semi-Ind"]["Avg Size"]["All"].append(
            avg_set_size(pred_sets, test_mask)
        )
        results[method][GNN_model]["Semi-Ind"]["Coverage"]["All"].append(cov)

        for t in range(T):
            # Consider test nodes only at time t
            if method == "BD":
                time_mask = np.array([[False] * n for _ in range(T)])
                time_mask[t] = True
                time_mask = time_mask.reshape(-1)
            if method == "UA":
                time_mask = np.array([[False] * n for _ in range(T + 1)])
                time_mask[t + 1] = True
                time_mask = time_mask.reshape(-1)

            test_mask_t = time_mask * test_mask
            if np.sum(test_mask_t) == 0:
                continue

            # Get prediction sets corresponding to time t
            pred_sets_t = pred_sets[
                np.array(
                    [
                        np.where(np.where(test_mask)[0] == np.where(test_mask_t)[0][i])[
                            0
                        ][0]
                        for i in range(sum(test_mask_t))
                    ]
                )
            ]

            results[method][GNN_model]["Semi-Ind"]["Accuracy"][t].append(
                accuracy(best_output, data, test_mask_t)
            )
            results[method][GNN_model]["Semi-Ind"]["Avg Size"][t].append(
                avg_set_size(pred_sets_t, test_mask_t)
            )
            results[method][GNN_model]["Semi-Ind"]["Coverage"][t].append(
                coverage(pred_sets_t, data, test_mask_t)
            )

# %% [markdown]
# Save results to pickle file.

# %%
with open(results_file, "wb") as file:
    pickle.dump(results, file)


# %%
