import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px
from scipy.linalg import block_diag
from scipy import sparse
from sklearn.decomposition import PCA
from tqdm import tqdm

import torch
from torch.functional import F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_scipy_sparse_matrix

from embedding_functions import general_unfolded_matrix


class Dynamic_Network(Dataset):
    """
    A pytorch geometric dataset for a dynamic network.
    """

    def __init__(self, As, node_labels, train_mask, val_mask, test_mask, x=None):
        """
        As: numpy array of adjacency matrices
        node_labels: (numerical) target labels for each node
        train_mask: boolean mask for training nodes
        test_mask: boolean mask for testing nodes
        """

        # check that As is sparse
        is_sparse = sparse.issparse(As[0])
        if is_sparse:
            self.As = np.array([sparse.csr_matrix(A) for A in As])
        else:
            self.As = As

        self.T = len(As)
        self.n = As[0].shape[0]

        self.node_labels = node_labels  # TODO: only for numerical targets currently
        assert len(node_labels) == self.n * self.T

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        assert len(self.train_mask) == self.n * self.T
        assert len(self.test_mask) == self.n * self.T

        # Option to assign each graph a feature vector
        if x is not None:
            self.x_all = x
            assert len(x) == self.n * self.T
        else:
            self.x_all = None

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):

        # If no features provided, just the identity matrix for each graph
        if self.x_all is None:
            x = torch.sparse.spdiags(
                torch.ones(self.n),
                offsets=torch.tensor([0]),
                shape=(self.n, self.n),
            )
        else:
            x = torch.tensor(
                self.x_all[self.n * idx : self.n * (idx + 1)], dtype=torch.float
            )

        edge_index = torch.tensor(
            np.array([self.As[idx].nonzero()]), dtype=torch.long
        ).reshape(2, -1)

        edge_weight = torch.tensor(np.array(self.As[idx].data), dtype=torch.float)

        classes = self.node_labels[self.n * idx : self.n * (idx + 1)]
        y = torch.tensor(classes)

        train_mask_for_idx = torch.tensor(
            self.train_mask[self.n * idx : self.n * (idx + 1)]
        )
        val_mask_for_idx = torch.tensor(
            self.val_mask[self.n * idx : self.n * (idx + 1)]
        )
        test_mask_for_idx = torch.tensor(
            self.test_mask[self.n * idx : self.n * (idx + 1)]
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=torch.randn(self.n, 1),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            train_mask=train_mask_for_idx,
            val_mask=val_mask_for_idx,
            test_mask=test_mask_for_idx,
        )
        data.num_nodes = self.n

        return data



class Block_Diagonal_Network(Dataset):
    """
    A pytorch geometric dataset for the block diagonal version of a Dynamic Network object.
    """

    def __init__(self, dataset, use_identity=False):

        self.check_sparse = sparse.issparse(dataset.As[0])

        if self.check_sparse:
            self.A = sparse.block_diag(dataset.As)
        else:
            self.A = block_diag(*dataset.As)

        self.T = dataset.T
        self.n = dataset.n
        self.y = torch.cat([data.y.clone().detach() for data in dataset]).reshape(
            (self.n * self.T,)
        )
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

        if not use_identity:
            self.x = torch.cat([data.x for data in dataset])
        else:
            self.x = torch.sparse.spdiags(
                torch.ones(self.n * (self.T)),
                offsets=torch.tensor([0]),
                shape=(self.n * (self.T), self.n * (self.T)),
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)

        edge_weight = torch.tensor(np.array(self.A.data), dtype=torch.float)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=self.x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=self.y,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask,
        )
        data.num_nodes = self.n * self.T

        return data


def block_GCN(
    As,
    node_labels,
    model_save_dir,
    train_mask,
    val_mask,
    test_mask,
    return_model=False,
    epoch_save_dir=None,
    num_epochs=201,
    hidden_channels=16,
):

    n = As[0].shape[0]

    # As we're doing classification make sure that the labels are long type
    node_labels = np.int64(node_labels)

    # default masks are the same for each time point
    if train_mask is None and test_mask is None:
        M = 2

        train_mask = [False] * n
        test_mask = [True] * n

        for label in np.unique(node_labels):
            for idx in np.random.choice(
                np.where(node_labels == label)[0], M, replace=False
            ):
                train_mask[idx] = True
                test_mask[idx] = False

        train_mask = np.tile(train_mask, len(As))
        test_mask = np.tile(test_mask, len(As))

    assert train_mask is not None
    assert test_mask is not None

    dataset = Dynamic_Network(
        As,
        node_labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    print("Using a single identity for block attributes")
    dataset_BD = Block_Diagonal_Network(dataset, use_identity=True)

    class GCN_BD(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset_BD.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset_BD.num_classes)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    def BD_train(model, data):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x, data.edge_index, data.edge_weight
        )  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask], data.y[data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def BD_val(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.val_mask] == data.y[data.val_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.val_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    def BD_test(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    model = GCN_BD(hidden_channels=hidden_channels)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters())  # Define optimizer.

    epoch_save_checkpoint = num_epochs // 50
    if epoch_save_dir is not None:
        print("Warning: this could use a lot of storage")

        if epoch_save_checkpoint == 0:
            epoch_save_checkpoint = 1

    print("Training Block GCN...")
    epoch_checkpoint = num_epochs // 10
    if epoch_checkpoint == 0:
        epoch_checkpoint = 1  # Stop division by zero if num_epochs < 10

    max_val_score = 0
    for epoch in tqdm(range(1, num_epochs + 1)):
        loss = BD_train(model, dataset_BD[0])
        val_acc = BD_val(model, dataset_BD[0])

        if val_acc > max_val_score:
            max_val_score = val_acc
            torch.save(model.state_dict(), model_save_dir)

        test_acc = BD_test(model, dataset_BD[0])

        if epoch_save_dir is not None:
            if epoch % epoch_save_checkpoint == 0:
                torch.save(model.state_dict(), epoch_save_dir + f"_{epoch}.pt")

        if epoch % epoch_checkpoint == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

    print("Training finished.")

    # load best model
    model.load_state_dict(torch.load(model_save_dir))
    best_model_acc = BD_test(model, dataset_BD[0])
    print("Best model accuracy: {}".format(best_model_acc))

    output_BD = (
        model(dataset_BD[0].x, dataset_BD[0].edge_index, dataset_BD[0].edge_weight)
        .detach()
        .numpy()
    )

    if return_model:
        return output_BD, model, best_model_acc
    else:
        return output_BD


class Unfolded_Network(Dataset):
    """
    A pytorch geometric dataset for the unfolding of a Dynamic Network object.
    """

    def __init__(self, dataset, use_identity=False):
        # check if As is sparse
        self.check_sparse = sparse.issparse(dataset.As[0])

        # If sparse, this function will return a sparse matrix
        self.A = general_unfolded_matrix(dataset.As, sparse_matrix=self.check_sparse)

        self.As = dataset.As
        self.T = dataset.T
        self.n = dataset.n
        self.y_dyn = torch.cat([data.y.clone().detach() for data in dataset]).reshape(
            (self.n * self.T,)
        )
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

        if not use_identity:
            x_dyn = torch.cat([data.x for data in dataset])

            x_anch = torch.sparse.spdiags(
                torch.ones(self.n),
                offsets=torch.tensor([0]),
                shape=(self.n, self.n),
            )
            # x_anch = torch.eye(self.n)

            assert len(x_anch) == self.n
            assert x_anch.shape[1] == x_dyn.shape[1]

            self.x = torch.cat((x_anch, x_dyn))
        else:
            self.x = torch.sparse.spdiags(
                torch.ones(self.n * (self.T + 1)),
                offsets=torch.tensor([0]),
                shape=(self.n * (self.T + 1), self.n * (self.T + 1)),
            )

        assert len(self.x) == self.n * (self.T + 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        # TODO: what to do here
        y_anch = torch.zeros(self.n, dtype=self.y_dyn.dtype).reshape((self.n,))

        y = torch.cat((y_anch, self.y_dyn))
        assert len(y) == self.n * (self.T + 1)

        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = torch.tensor(np.array(self.A.data), dtype=torch.float)

        # Do not include anchor nodes in training or test
        train_mask = torch.tensor(
            np.concatenate([np.repeat(False, self.n), self.train_mask])
        )
        val_mask = torch.tensor(
            np.concatenate([np.repeat(False, self.n), self.val_mask])
        )
        test_mask = torch.tensor(
            np.concatenate([np.repeat(False, self.n), self.test_mask])
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=self.x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        data.num_nodes = self.n * (self.T + 1)

        return data


def UGCN(
    As,
    node_labels,
    model_save_dir,
    train_mask,
    val_mask,
    test_mask,
    return_model=False,
    epoch_save_dir=None,
    num_epochs=201,
    hidden_channels=16,
):

    n = As[0].shape[0]
    T = len(As)

    # Make sure As is sparse
    is_sparse = sparse.issparse(As[0])
    if is_sparse:
        As = [sparse.csr_matrix(A) for A in As]

    # As we're doing classification make sure that the labels are long type
    node_labels = np.int64(node_labels)

    assert len(node_labels) == n * len(As)

    dataset = Dynamic_Network(
        As,
        node_labels,
        train_mask,
        val_mask,
        test_mask,
    )
    print("Using a single identity for unfolded attributes")
    dataset_UA = Unfolded_Network(dataset, use_identity=True)

    class GCN_UA(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset_UA.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset_UA.num_classes)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    def UGCN_train(model, data):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x, data.edge_index, data.edge_weight
        )  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask],
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def UGCN_val(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.val_mask] == data.y[data.val_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.val_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    def UGCN_test(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    model = GCN_UA(hidden_channels=hidden_channels)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

    optimizer = torch.optim.Adam(model.parameters())  # Define optimizer.

    epoch_save_checkpoint = num_epochs // 50
    if epoch_save_dir is not None:
        print("Warning: this could use a lot of storage")

        if epoch_save_checkpoint == 0:
            epoch_save_checkpoint = 1


    print("Training Unfolded GCN...")
    max_val_score = 0
    epoch_checkpoint = num_epochs // 10
    if epoch_checkpoint == 0:
        epoch_checkpoint = 1  # Stop division by zero if num_epochs < 10

    for epoch in tqdm(range(1, num_epochs + 1)):
        loss = UGCN_train(model, dataset_UA[0])
        val_acc = UGCN_val(model, dataset_UA[0])

        if val_acc > max_val_score:
            max_val_score = val_acc
            torch.save(model.state_dict(), model_save_dir)

        test_acc = UGCN_test(model, dataset_UA[0])

        if epoch_save_dir is not None:
            if epoch % epoch_save_checkpoint == 0:
                torch.save(model.state_dict(), epoch_save_dir + f"_{epoch}.pt")

        if epoch % epoch_checkpoint == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

    print("Training finished.")

    # load best model
    model.load_state_dict(torch.load(model_save_dir))
    best_model_acc = UGCN_test(model, dataset_UA[0])
    print("Best model accuracy: {}".format(best_model_acc))

    output_UA = (
        model(dataset_UA[0].x, dataset_UA[0].edge_index, dataset_UA[0].edge_weight)
        .detach()
        .numpy()
    )

    right = output_UA[n:, :]
    left = output_UA[:n, :]

    if return_model:
        return right, model, best_model_acc

    else:
        return right


# def UGCN_regression_legacy(
#     As,
#     node_labels,
#     node_attributes=None,
#     train_mask=None,
#     test_mask=None,
#     return_model=False,
#     num_epochs=201,
#     hidden_channels=16,
# ):

#     n = As[0].shape[0]
#     T = len(As)

#     # As we're doing regression make sure that the labels are float type
#     node_labels = np.float64(node_labels)

#     # default masks are selected randomly
#     if train_mask is None and test_mask is None:
#         train_test_ratio = 0.1
#         train_mask = np.random.choice(
#             [True, False], size=n * T, p=[train_test_ratio, 1 - train_test_ratio]
#         )
#         test_mask = ~train_mask

#     assert train_mask is not None
#     assert test_mask is not None
#     assert len(node_labels) == n * len(As)

#     if node_attributes is None:
#         dyn_node_attributes = None
#         anch_node_attributes = None
#     else:
#         assert node_attributes.shape[0] == n * (len(As) + 1)

#         dyn_node_attributes = node_attributes[n:, :].reshape((n * len(As), -1))
#         anch_node_attributes = node_attributes[:n, :].reshape((n, -1))

#     print("Removing dyn node attributes for now")
#     dataset = Dynamic_Network(
#         As,
#         node_labels,
#         train_mask,
#         test_mask,  # x=dyn_node_attributes
#     )
#     print("Using a single identity for unfolded attributes")
#     dataset_UA = Unfolded_Network(dataset, use_identity=True)

#     class GCN_UA(torch.nn.Module):
#         def __init__(self, hidden_channels):
#             super().__init__()
#             torch.manual_seed(1234567)
#             self.conv1 = GCNConv(dataset_UA.num_features, hidden_channels)
#             # self.conv2 = GCNConv(hidden_channels, dataset_UA.num_features)
#             self.conv2 = GCNConv(hidden_channels, 1)
#             # self.linear1 = torch.nn.Linear(dataset_UA.num_features, 1)

#         def forward(self, x, edge_index, edge_weight):
#             x = self.conv1(x, edge_index, edge_weight)
#             x = x.relu()
#             x = F.dropout(x, p=0.5, training=self.training)
#             x = self.conv2(x, edge_index, edge_weight)
#             # x = self.linear1(x)
#             return x

#     def UGCN_train(data):
#         # data.x = data.x.float()
#         # # data.edge_index = data.edge_index.float()
#         # data.edge_weight = data.edge_weight.float()
#         # data.y = data.y.float()

#         model.train()
#         optimizer.zero_grad()  # Clear gradients.
#         out = model(
#             data.x, data.edge_index, data.edge_weight
#         )  # Perform a single forward pass.
#         loss = criterion(
#             out[data.train_mask].reshape(-1),
#             data.y[data.train_mask],
#         )  # Compute the loss solely based on the training nodes.
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         return loss

#     def UGCN_test(data):
#         model.eval()
#         out = model(data.x, data.edge_index, data.edge_weight)
#         # Calculate the RMSE
#         mse = criterion(out[data.test_mask].reshape(-1), data.y[data.test_mask])
#         rmse = torch.sqrt(mse)
#         return rmse.item()

#     model = GCN_UA(hidden_channels=hidden_channels)

#     # criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
#     criterion = torch.nn.MSELoss()

#     optimizer = torch.optim.Adam(model.parameters())  # Define optimizer.

#     print("Training UGCN...")
#     epoch_checkpoint = num_epochs // 10
#     if epoch_checkpoint == 0:
#         epoch_checkpoint = 1  # Stop division by zero if num_epochs < 10

#     for epoch in tqdm(range(1, num_epochs)):
#         loss = UGCN_train(dataset_UA[0])
#         rmse = UGCN_test(dataset_UA[0])

#         if epoch % epoch_checkpoint == 0:
#             print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, RMSE: {rmse:.4f}")

#     print(f"Training finished. RMSE: {UGCN_test(dataset_UA[0]):.4f}")
#     output_UA = (
#         model(dataset_UA[0].x, dataset_UA[0].edge_index, dataset_UA[0].edge_weight)
#         .detach()
#         .numpy()
#     )

#     right = output_UA[n:, :]
#     left = output_UA[:n, :]

#     if return_model:
#         return right, model

#     else:
#         return right


def UGCN_regression(
    As,
    node_labels,
    node_attributes=None,
    train_mask=None,
    test_mask=None,
    return_model=False,
    num_epochs=201,
    hidden_channels=16,
):

    n = As[0].shape[0]
    T = len(As)

    # As we're doing regression make sure that the labels are float type
    node_labels = np.float64(node_labels)

    # default masks are selected randomly
    if train_mask is None and test_mask is None:
        train_test_ratio = 0.1
        train_mask = np.random.choice(
            [True, False], size=n * T, p=[train_test_ratio, 1 - train_test_ratio]
        )
        test_mask = ~train_mask

    assert train_mask is not None
    assert test_mask is not None
    assert len(node_labels) == n * len(As)

    if node_attributes is None:
        dyn_node_attributes = None
        anch_node_attributes = None
    else:
        assert node_attributes.shape[0] == n * (len(As) + 1)

        dyn_node_attributes = node_attributes[n:, :].reshape((n * len(As), -1))
        anch_node_attributes = node_attributes[:n, :].reshape((n, -1))

    print("Removing dyn node attributes for now")
    dataset = Dynamic_Network(
        As,
        node_labels,
        train_mask,
        test_mask,  # x=dyn_node_attributes
    )
    print("Using a single identity for unfolded attributes")
    dataset_UA = Unfolded_Network(dataset, use_identity=True)

    class GCN_UA(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset_UA.num_features, hidden_channels)
            # self.conv2 = GCNConv(hidden_channels, dataset_UA.num_features)
            self.conv2 = GCNConv(hidden_channels, 1)
            # self.linear1 = torch.nn.Linear(dataset_UA.num_features, 1)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            # x = self.linear1(x)
            return x

    def UGCN_train(data):
        # data.x = data.x.float()
        # # data.edge_index = data.edge_index.float()
        # data.edge_weight = data.edge_weight.float()
        # data.y = data.y.float()

        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x, data.edge_index, data.edge_weight
        )  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask].reshape(-1),
            data.y[data.train_mask],
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def UGCN_test(data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        # Calculate the RMSE
        mse = criterion(out[data.test_mask].reshape(-1), data.y[data.test_mask])
        rmse = torch.sqrt(mse)
        return rmse.item()

    model = GCN_UA(hidden_channels=hidden_channels)

    # criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())  # Define optimizer.

    print("Training UGCN...")
    epoch_checkpoint = num_epochs // 10
    if epoch_checkpoint == 0:
        epoch_checkpoint = 1  # Stop division by zero if num_epochs < 10

    for epoch in tqdm(range(1, num_epochs)):
        loss = UGCN_train(dataset_UA[0])
        rmse = UGCN_test(dataset_UA[0])

        if epoch % epoch_checkpoint == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, RMSE: {rmse:.4f}")

    print(f"Training finished. RMSE: {UGCN_test(dataset_UA[0]):.4f}")
    output_UA = (
        model(dataset_UA[0].x, dataset_UA[0].edge_index, dataset_UA[0].edge_weight)
        .detach()
        .numpy()
    )

    right = output_UA[n:, :]
    left = output_UA[:n, :]

    if return_model:
        return right, model

    else:
        return right


def block_GAT(
    As,
    node_labels,
    model_save_dir,
    train_mask,
    val_mask,
    test_mask,
    return_model=False,
    epoch_save_dir=None,
    num_epochs=201,
    hidden_channels=16,
):

    n = As[0].shape[0]

    # As we're doing classification make sure that the labels are long type
    node_labels = np.int64(node_labels)

    # default masks are the same for each time point
    if train_mask is None and test_mask is None:
        M = 2

        train_mask = [False] * n
        test_mask = [True] * n

        for label in np.unique(node_labels):
            for idx in np.random.choice(
                np.where(node_labels == label)[0], M, replace=False
            ):
                train_mask[idx] = True
                test_mask[idx] = False

        train_mask = np.tile(train_mask, len(As))
        test_mask = np.tile(test_mask, len(As))

    assert train_mask is not None
    assert test_mask is not None

    dataset = Dynamic_Network(
        As,
        node_labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    print("Using a single identity for block attributes")
    dataset_BD = Block_Diagonal_Network(dataset, use_identity=True)

    class GAT_BD(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GATConv(dataset_BD.num_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, dataset_BD.num_classes)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    def BD_train(model, data):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x, data.edge_index, data.edge_weight
        )  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask], data.y[data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def BD_val(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.val_mask] == data.y[data.val_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.val_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    def BD_test(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    model = GAT_BD(hidden_channels=hidden_channels)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters())  # Define optimizer.

    epoch_save_checkpoint = num_epochs // 50
    if epoch_save_dir is not None:
        print("Warning: this could use a lot of storage")

        if epoch_save_checkpoint == 0:
            epoch_save_checkpoint = 1

    print("Training Block GAT...")
    epoch_checkpoint = num_epochs // 10
    if epoch_checkpoint == 0:
        epoch_checkpoint = 1  # Stop division by zero if num_epochs < 10

    max_val_score = 0
    for epoch in tqdm(range(1, num_epochs + 1)):
        loss = BD_train(model, dataset_BD[0])
        val_acc = BD_val(model, dataset_BD[0])

        if val_acc > max_val_score:
            max_val_score = val_acc
            torch.save(model.state_dict(), model_save_dir)

        test_acc = BD_test(model, dataset_BD[0])

        if epoch_save_dir is not None:
            if epoch % epoch_save_checkpoint == 0:
                torch.save(model.state_dict(), epoch_save_dir + f"_{epoch}.pt")

        if epoch % epoch_checkpoint == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

    print("Training finished.")

    # load best model
    model.load_state_dict(torch.load(model_save_dir))
    best_model_acc = BD_test(model, dataset_BD[0])
    print("Best model accuracy: {}".format(best_model_acc))

    output_BD = (
        model(dataset_BD[0].x, dataset_BD[0].edge_index, dataset_BD[0].edge_weight)
        .detach()
        .numpy()
    )

    if return_model:
        return output_BD, model, best_model_acc
    else:
        return output_BD

def UGAT(
    As,
    node_labels,
    model_save_dir,
    train_mask,
    val_mask,
    test_mask,
    return_model=False,
    epoch_save_dir=None,
    num_epochs=201,
    hidden_channels=16,
):

    n = As[0].shape[0]
    T = len(As)

    # Make sure As is sparse
    is_sparse = sparse.issparse(As[0])
    if is_sparse:
        As = [sparse.csr_matrix(A) for A in As]

    # As we're doing classification make sure that the labels are long type
    node_labels = np.int64(node_labels)

    assert len(node_labels) == n * len(As)

    dataset = Dynamic_Network(
        As,
        node_labels,
        train_mask,
        val_mask,
        test_mask,
    )
    print("Using a single identity for unfolded attributes")
    dataset_UA = Unfolded_Network(dataset, use_identity=True)

    class GAT_UA(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GATConv(dataset_UA.num_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, dataset_UA.num_classes)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    def UGAT_train(model, data):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x, data.edge_index, data.edge_weight
        )  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask],
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def UGAT_val(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.val_mask] == data.y[data.val_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.val_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    def UGAT_test(model, data):
        model.eval()
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return acc

    model = GAT_UA(hidden_channels=hidden_channels)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

    optimizer = torch.optim.Adam(model.parameters())  # Define optimizer.

    epoch_save_checkpoint = num_epochs // 50
    if epoch_save_dir is not None:
        print("Warning: this could use a lot of storage")

        if epoch_save_checkpoint == 0:
            epoch_save_checkpoint = 1


    print("Training Unfolded GAT...")
    max_val_score = 0
    epoch_checkpoint = num_epochs // 10
    if epoch_checkpoint == 0:
        epoch_checkpoint = 1  # Stop division by zero if num_epochs < 10

    for epoch in tqdm(range(1, num_epochs + 1)):
        loss = UGAT_train(model, dataset_UA[0])
        val_acc = UGAT_val(model, dataset_UA[0])

        if val_acc > max_val_score:
            max_val_score = val_acc
            torch.save(model.state_dict(), model_save_dir)

        test_acc = UGAT_test(model, dataset_UA[0])

        if epoch_save_dir is not None:
            if epoch % epoch_save_checkpoint == 0:
                torch.save(model.state_dict(), epoch_save_dir + f"_{epoch}.pt")

        if epoch % epoch_checkpoint == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

    print("Training finished.")

    # load best model
    model.load_state_dict(torch.load(model_save_dir))
    best_model_acc = UGAT_test(model, dataset_UA[0])
    print("Best model accuracy: {}".format(best_model_acc))

    output_UA = (
        model(dataset_UA[0].x, dataset_UA[0].edge_index, dataset_UA[0].edge_weight)
        .detach()
        .numpy()
    )

    right = output_UA[n:, :]
    left = output_UA[:n, :]

    if return_model:
        return right, model, best_model_acc

    else:
        return right
