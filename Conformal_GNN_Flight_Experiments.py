# %%
import copy
from itertools import product
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch
from torch.functional import F
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, GCNConv
from scipy import sparse

# Specifically for the flight data processing
import pycountry

# %% [markdown]
# ## Experiment parameters

# %%
dataset_name = "Flight"


# Training/validation/calibration/test dataset split sizes
props = np.array([0.2, 0.1, 0.35, 0.35])

# Target 1-coverage for conformal prediction
alpha = 0.1

# Number of experiments
num_train_trans = 10
num_permute_trans = 100
num_train_semi_ind = 50

# GNN model parameters
num_epochs = 30
num_channels_GCN = 32
num_channels_GAT = 32


# Save results
results_file = f"results/Conformal_GNN_{dataset_name}_Results.pkl"

# %% [markdown]
# ## Load dataset

# %%
print("Processing the flight data...")

datapath = "datasets/flight_data/"

# Load adjacency matrices
As = []
T = 36
for t in range(T):
    As.append(sparse.load_npz(datapath + "As_" + str(t) + ".npz"))

As = np.array(As)
n = As[0].shape[0]

# Load labels
Z = np.load(datapath + "Z.npy")
nodes = np.load(datapath + "nodes.npy", allow_pickle=True).item()


months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
labels = np.core.defchararray.add(
    np.array(months * 3), np.repeat(["2019", "2020", "2021"], 12)
)

euro_nodes_idx = np.where(np.array(Z) == "EU")[0]

airports = pd.read_csv(datapath + "airports.csv")
country_codes_in_data = []
airports_not_found = []
flights_for_each_country_month = []
flights_for_each_country_number = []
flights_for_each_country_airport = []
flights_for_each_country_code = []
flights_for_each_country_country = []
for code in tqdm(np.array(list(nodes.keys()))[euro_nodes_idx]):
    try:
        country_of_airport = airports[airports["ident"] == code]["iso_country"].values[
            0
        ]
        country_codes_in_data.append(country_of_airport)

        for t in range(T):
            flights_for_each_country_month.append(labels[t])
            flights_for_each_country_number.append(np.sum(As[t][:, nodes[code]]))
            flights_for_each_country_airport.append(code)
            flights_for_each_country_code.append(country_of_airport)
            try:
                country_name = pycountry.countries.get(alpha_2=country_of_airport).name
                flights_for_each_country_country.append(country_name)
            except:
                if country_of_airport == "XK":
                    flights_for_each_country_country.append("Kosovo")
                else:
                    print("can't find country for code: ", country_of_airport)
                    flights_for_each_country_country.append("Unknown")
    except:
        airports_not_found.append(code)

len(airports_not_found)

flights_for_each_country = pd.DataFrame(
    {
        "month": flights_for_each_country_month,
        "airport": flights_for_each_country_airport,
        "number_of_flights": flights_for_each_country_number,
        "country_code": flights_for_each_country_code,
        "country": flights_for_each_country_country,
    }
)

flights_for_each_country.loc[
    flights_for_each_country["country"] == "Unknown", "country"
] = "Kosovo"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Czechia", "country"
] = "Czech Republic"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Slovakia", "country"
] = "Slovak Republic"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Russian Federation", "country"
] = "Russia"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Moldova, Republic of", "country"
] = "Moldova"

# %%
euro_airports = np.array(list(nodes.keys()))[euro_nodes_idx]
airport_to_country = dict(
    zip(flights_for_each_country["airport"], flights_for_each_country["country"])
)
country_labels = np.array([airport_to_country[code] for code in euro_airports])
# %%

# Set the target variable to be the continent (Z) and encode them
node_labels = Z
node_labels_enc = pd.Categorical(node_labels)
node_labels = np.tile(np.array(node_labels_enc.codes), T)


# %%

As_euro = []
for A in As:
    A_euro = A[euro_nodes_idx, :][:, euro_nodes_idx]
    As_euro.append(A_euro)

n = As_euro[0].shape[0]
node_labels_enc = pd.Categorical(country_labels)
node_labels = np.tile(node_labels_enc.codes, T)
As = As_euro


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
data_mask = np.array([[True] * T for _ in range(n)])

for t in range(T):
    data_mask[np.where(np.sum(As[t], axis=0) == 0)[0], t] = False
    # data_mask[np.where(node_labels == label_dict['Teachers'])[0], t] = False

num_classes = np.unique(node_labels).shape[0]
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

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask)
            valid_acc = valid(model, data, valid_mask)

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
