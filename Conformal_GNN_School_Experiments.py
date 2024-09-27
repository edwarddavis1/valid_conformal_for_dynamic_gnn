# %%
import copy
from itertools import product
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

# %% [markdown]
# ## Experiment parameters

# %%
# Training/validation/calibration/test dataset split sizes
props = np.array([0.2, 0.1, 0.35, 0.35])

assert np.sum(props) == 1

# Target 1-coverage for conformal prediction
alpha = 0.1

# print("WARNING: reduced parameters")
# num_train_trans = 2
# num_permute_trans = 1
# num_train_semi_ind = 2
# num_epochs = 10

# Number of experiments
num_train_trans = 10
num_permute_trans = 100
num_train_semi_ind = 50

# GNN model parameters
num_epochs = 500
num_channels_GCN = 16
num_channels_GAT = 16
learning_rate = 0.01
weight_decay = 5e-4

# Save results
# results_file = 'results/Conformal_GNN_School_Results_10_100_50.pkl'
results_file = "results/School_with_assisted_semi_ind.pkl"
# results_file = "results/school_with_5perc_training.pkl"

# %% [markdown]
# ## Load dataset

# %%
window = 60 * 60

day_1_start = (8 * 60 + 30) * 60
day_1_end = (17 * 60 + 30) * 60
day_2_start = ((24 + 8) * 60 + 30) * 60
day_2_end = ((24 + 17) * 60 + 30) * 60

T1 = int((day_1_end - day_1_start) // window)
T2 = int((day_2_end - day_2_start) // window)
T = T1 + T2

print(f"Number of time windows: {T}")

# %%
fname = "datasets/ia-primary-school-proximity-attr.edges"
file = open(fname)

label_dict = {
    "1A": 0,
    "1B": 1,
    "2A": 2,
    "2B": 3,
    "3A": 4,
    "3B": 5,
    "4A": 6,
    "4B": 7,
    "5A": 8,
    "5B": 9,
    "Teachers": 10,
}
num_classes = 10

nodes = []
node_labels = []
edge_tuples = []

for line in file:
    node_i, node_j, time, id_i, id_j = line.strip("\n").split(",")

    if day_1_start <= int(time) < day_1_end:
        t = (int(time) - day_1_start) // window
    elif day_2_start <= int(time) < day_2_end:
        t = T1 + (int(time) - day_2_start) // window
    else:
        continue

    if node_i not in nodes:
        nodes.append(node_i)
        node_labels.append(label_dict[id_i])

    if node_j not in nodes:
        nodes.append(node_j)
        node_labels.append(label_dict[id_j])

    edge_tuples.append([t, node_i, node_j])

edge_tuples = np.unique(edge_tuples, axis=0)
nodes = np.array(nodes)

n = len(nodes)
print(f"Number of nodes: {n}")

node_dict = dict(zip(nodes[np.argsort(node_labels)], range(n)))
node_labels = np.sort(node_labels)

# %% [markdown]
# Create a list of adjacency matrices.

# %%
As = np.zeros((T, n, n))

for m in range(len(edge_tuples)):
    t, i, j = edge_tuples[m]
    As[int(t), node_dict[i], node_dict[j]] = 1
    As[int(t), node_dict[j], node_dict[i]] = 1

# %% [markdown]
# ## GNN functions

# %% [markdown]
# Dynamic network class that puts a list of adjacency matrices along with class labels into a pytorch geometric dataset. This is then used to create the block diagonal and dilated unfolded adjacency matrices for input into the graph neural networks.


# %%
class Dynamic_Network(Dataset):
    """
    A pytorch geometric dataset for a dynamic network.
    """

    def __init__(self, As, labels):
        self.As = As
        self.T = As.shape[0]
        self.n = As.shape[1]
        self.classes = labels

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):
        x = torch.tensor(np.eye(self.n), dtype=torch.float)
        edge_index = torch.tensor(
            np.array([self.As[idx].nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        y = torch.tensor(self.classes, dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = self.n

        return data


# %%
class Block_Diagonal_Network(Dataset):
    """
    A pytorch geometric dataset for the block diagonal version of a Dynamic Network object.
    """

    def __init__(self, dataset):
        self.A = block_diag(*dataset.As)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.tensor(np.eye(self.n * self.T), dtype=torch.float)
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        y = torch.tensor(np.tile(self.classes, self.T), dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = self.n * self.T

        return data


# %%
class Unfolded_Network(Dataset):
    """
    A pytorch geometric dataset for the dilated unfolding of a Dynamic Network object.
    """

    def __init__(self, dataset):
        self.A_unf = np.block([dataset.As[t] for t in range(dataset.T)])
        self.A = np.block(
            [
                [np.zeros((dataset.n, dataset.n)), self.A_unf],
                [
                    self.A_unf.T,
                    np.zeros((dataset.n * dataset.T, dataset.n * dataset.T)),
                ],
            ]
        )
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.tensor(np.eye(self.n * (self.T + 1)), dtype=torch.float)
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        y = torch.tensor(np.tile(self.classes, (self.T + 1)), dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
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

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GAT(torch.nn.Module):
    def __init__(self, num_nodes, num_channels, num_classes, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GATConv(num_nodes, num_channels)
        self.conv2 = GATConv(num_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x


# %%
def train(model, data, train_mask):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()

    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    return loss


def valid(model, data, valid_mask):
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[valid_mask] == data.y[valid_mask]
    acc = int(correct.sum()) / int(valid_mask.sum())

    return acc


# %% [markdown]
# ## Data split functions

# %% [markdown]
# Create a data mask that consists only of useful nodes for training and testing. We only consider nodes in a particular time window if it has degree greater than zero at that time window. Also, we ignore nodes with label `Teacher` meaning that the number of classes is now 10.

# %%
data_mask = np.array([[True] * T for _ in range(n)])

for t in range(T):
    data_mask[np.where(np.sum(As[t], axis=0) == 0)[0], t] = False
    data_mask[np.where(node_labels == label_dict["Teachers"])[0], t] = False

num_classes = 10
print(
    f"Percentage of usable node/time pairs: {100 * np.sum(data_mask) / (n * T) :02.1f}%"
)


# %%
def mask_split(mask, split_props, seed=0, mode="transductive"):
    np.random.seed(seed)

    n, T = mask.shape

    if mode == "transductive":
        # Flatten mask array into one dimension in blocks of nodes per time
        flat_mask = mask.T.reshape(-1)
        n_masks = np.sum(flat_mask)

        # Split shuffled flatten mask array indices into correct proportions
        flat_mask_idx = np.where(flat_mask)[0]
        np.random.shuffle(flat_mask_idx)
        split_ns = np.cumsum([round(n_masks * prop) for prop in split_props[:-1]])
        split_idx = np.split(flat_mask_idx, split_ns)

    if mode == "semi-inductive":

        # Find time such that final proportion of masks happen after that time
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        # Flatten mask arrays into one dimension in blocks of nodes per time
        flat_mask_start = mask[:, :T_trunc].T.reshape(-1)
        flat_mask_end = mask[:, T_trunc:].T.reshape(-1)
        n_masks_start = np.sum(flat_mask_start)

        # Split starting shuffled flatten mask array into correct proportions
        flat_mask_start_idx = np.where(flat_mask_start)[0]
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-1] / np.sum(split_props[:-1])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)

        # Place finishing flatten mask array at the end
        split_idx.append(n * T_trunc + np.where(flat_mask_end)[0])

    if mode == "assisted semi-inductive":

        # Find time such that final proportion of masks happen after that time
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        # Flatten mask arrays into one dimension in blocks of nodes per time
        flat_mask_start = mask[:, :T_trunc].T.reshape(-1)
        flat_mask_end = mask[:, T_trunc:].T.reshape(-1)
        n_masks_start = np.sum(flat_mask_start)

        # Split starting shuffled flatten mask array into correct proportions
        flat_mask_start_idx = np.where(flat_mask_start)[0]
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-2] / np.sum(split_props[:-2])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)

        # Do the same for after T_trunc
        flat_mask_end_idx = np.where(flat_mask_end)[0]
        np.random.shuffle(flat_mask_end_idx)
        split_props_end = split_props[-2:] / np.sum(split_props[-2:])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_end[:-1]]
        )
        split_idx.append(n * T_trunc + np.split(flat_mask_end_idx, split_ns)[0])
        split_idx.append(n * T_trunc + np.split(flat_mask_end_idx, split_ns)[1])

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
train_mask, valid_mask, calib_mask, test_mask = mask_split(
    data_mask, props, mode="assisted semi-inductive"
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
regimes = ["Assisted Semi-Ind", "Trans", "Semi-Ind"]
# regimes = ["Trans", "Semi-Ind"]
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
# ### Assisted Semi-inductive experiments

# %%

for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_trans):
        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, mode="assisted semi-inductive"
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

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        print(f"\nTraining {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask)
            valid_acc = valid(model, data, valid_mask)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        print(f"Evaluating {method_str} {GNN_model} Number {i}")
        print(f"Validation accuracy: {max_valid_acc:0.3f}")
        output = best_model(data.x, data.edge_index)

        for j in tqdm(range(num_permute_trans)):
            # Permute the calibration and test datasets
            calib_mask, test_mask = mask_mix(calib_mask, test_mask, seed=j)

            pred_sets = get_prediction_sets(output, data, calib_mask, test_mask, alpha)

            results[method][GNN_model]["Assisted Semi-Ind"]["Accuracy"]["All"].append(
                accuracy(output, data, test_mask)
            )
            results[method][GNN_model]["Assisted Semi-Ind"]["Avg Size"]["All"].append(
                avg_set_size(pred_sets, test_mask)
            )
            results[method][GNN_model]["Assisted Semi-Ind"]["Coverage"]["All"].append(
                coverage(pred_sets, data, test_mask)
            )

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
                            np.where(
                                np.where(test_mask)[0] == np.where(test_mask_t)[0][i]
                            )[0][0]
                            for i in range(sum(test_mask_t))
                        ]
                    )
                ]

                results[method][GNN_model]["Assisted Semi-Ind"]["Accuracy"][t].append(
                    accuracy(output, data, test_mask_t)
                )
                results[method][GNN_model]["Assisted Semi-Ind"]["Avg Size"][t].append(
                    avg_set_size(pred_sets_t, test_mask_t)
                )
                results[method][GNN_model]["Assisted Semi-Ind"]["Coverage"][t].append(
                    coverage(pred_sets_t, data, test_mask_t)
                )

        avg_test_acc = np.mean(
            results[method][GNN_model]["Assisted Semi-Ind"]["Accuracy"]["All"]
        )
        print(f"Test accuracy: {avg_test_acc:0.3f}")

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

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        print(f"\nTraining {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask)
            valid_acc = valid(model, data, valid_mask)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        print(f"Evaluating {method_str} {GNN_model} Number {i}")
        print(f"Validation accuracy: {max_valid_acc:0.3f}")
        output = best_model(data.x, data.edge_index)

        for j in tqdm(range(num_permute_trans)):
            # Permute the calibration and test datasets
            calib_mask, test_mask = mask_mix(calib_mask, test_mask, seed=j)

            pred_sets = get_prediction_sets(output, data, calib_mask, test_mask, alpha)

            results[method][GNN_model]["Trans"]["Accuracy"]["All"].append(
                accuracy(output, data, test_mask)
            )
            results[method][GNN_model]["Trans"]["Avg Size"]["All"].append(
                avg_set_size(pred_sets, test_mask)
            )
            results[method][GNN_model]["Trans"]["Coverage"]["All"].append(
                coverage(pred_sets, data, test_mask)
            )

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
                            np.where(
                                np.where(test_mask)[0] == np.where(test_mask_t)[0][i]
                            )[0][0]
                            for i in range(sum(test_mask_t))
                        ]
                    )
                ]

                results[method][GNN_model]["Trans"]["Accuracy"][t].append(
                    accuracy(output, data, test_mask_t)
                )
                results[method][GNN_model]["Trans"]["Avg Size"][t].append(
                    avg_set_size(pred_sets_t, test_mask_t)
                )
                results[method][GNN_model]["Trans"]["Coverage"][t].append(
                    coverage(pred_sets_t, data, test_mask_t)
                )

        avg_test_acc = np.mean(results[method][GNN_model]["Trans"]["Accuracy"]["All"])
        print(f"Test accuracy: {avg_test_acc:0.3f}")

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

        optimizer = torch.optim.Adam(
            model.parameters()
        )  # , lr=learning_rate, weight_decay=weight_decay)

        print(f"\nTraining {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        val_curve = []
        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask)
            valid_acc = valid(model, data, valid_mask)
            val_curve.append(valid_acc)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)
                best_epoch = epoch

        print(f"Evaluating {method_str} {GNN_model} Number {i}")
        print(f"Validation accuracy: {max_valid_acc:0.3f}")
        print(f"Best epoch: {best_epoch}")
        output = best_model(data.x, data.edge_index)

        # Cannot permute the calibration and test datasets in semi-inductive experiments

        pred_sets = get_prediction_sets(output, data, calib_mask, test_mask, alpha)

        results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"].append(
            accuracy(output, data, test_mask)
        )
        results[method][GNN_model]["Semi-Ind"]["Avg Size"]["All"].append(
            avg_set_size(pred_sets, test_mask)
        )
        results[method][GNN_model]["Semi-Ind"]["Coverage"]["All"].append(
            coverage(pred_sets, data, test_mask)
        )

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
                accuracy(output, data, test_mask_t)
            )
            results[method][GNN_model]["Semi-Ind"]["Avg Size"][t].append(
                avg_set_size(pred_sets_t, test_mask_t)
            )
            results[method][GNN_model]["Semi-Ind"]["Coverage"][t].append(
                coverage(pred_sets_t, data, test_mask_t)
            )

        avg_test_acc = np.mean(
            results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"]
        )
        print(f"Test accuracy: {avg_test_acc:0.3f}")

# %% [markdown]
# Save results to pickle file.

# %%
with open(results_file, "wb") as file:
    pickle.dump(results, file)


# %%
