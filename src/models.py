"""GNN model definitions: GCN, GIN, GAT.

All three architectures share the same pattern:
1. Embedding / projection layer
2. N message-passing layers with BatchNorm + dropout
3. Graph-level readout (pooling)
4. Classification head (MLP → single logit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    SAGEConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
)


class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017).

    Aggregation: normalized mean of neighbor features.
    Simplest GNN baseline — tests whether basic neighborhood averaging captures
    enough local structure when the model sees the explicit graph.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019).

    Aggregation: sum + MLP update.
    Theoretically the most expressive GNN — as powerful as the 1-WL
    isomorphism test. Uses sum (not mean) to distinguish multisets.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        return self.head(x).squeeze(-1)


class GAT(nn.Module):
    """Graph Attention Network (Veličković et al., 2018).

    Aggregation: attention-weighted sum of neighbor features.
    Multi-head attention lets the model learn which neighbor atoms
    are most important for predicting the target property.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Last layer: single head for clean output
        self.convs.append(
            GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)  # ELU is standard for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


class GraphSAGE(nn.Module):
    """GraphSAGE (Hamilton et al., 2017).

    Aggregation: concat(self, mean(neighbours)).
    Designed for inductive learning on unseen graphs.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


class GINe(nn.Module):
    """GIN with edge features (Hu et al., 2020).

    Same as GIN but bond-level features (type, conjugation, ring membership)
    are injected via GINEConv. Tests whether bond information improves
    over topology-only GIN.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.3, edge_dim=6):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        edge_attr = self.edge_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        return self.head(x).squeeze(-1)
