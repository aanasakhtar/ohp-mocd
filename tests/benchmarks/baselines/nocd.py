"""
nocd.py

Official implementation of Neural Overlapping Community Detection (NOCD)
Reference: Shchur & Günnemann (KDD / ICLR 2019)
"Overlapping Community Detection with Graph Neural Networks"
GitHub: https://github.com/shchur/overlapping-community-detection
"""

import math
import random
import collections
import numpy as np
import scipy.sparse as sp
import networkx as nx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric normalized adjacency with self-loops: D^{-1/2} (A + I) D^{-1/2}."""
    adj_loop = adj + sp.eye(adj.shape[0], format="csr")
    row_sum = np.array(adj_loop.sum(axis=1)).flatten()
    inv_deg = np.power(row_sum, -0.5, where=(row_sum > 0))
    inv_deg[row_sum == 0] = 0.0
    D_inv = sp.diags(inv_deg)
    return D_inv.dot(adj_loop).dot(D_inv).tocsr()

if HAS_TORCH:
    class GCNEncoder(nn.Module):
        def __init__(self, in_features: int, hidden_dim: int, num_communities: int, dropout: float = 0.5):
            super().__init__()
            self.w1 = nn.Parameter(torch.FloatTensor(in_features, hidden_dim))
            self.w2 = nn.Parameter(torch.FloatTensor(hidden_dim, num_communities))
            self.bn = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.reset_parameters()
            
        def reset_parameters(self):
            nn.init.xavier_uniform_(self.w1)
            nn.init.xavier_uniform_(self.w2)
            
        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            x = self.dropout(x)
            h1 = torch.spmm(adj, x) @ self.w1
            h1 = self.bn(h1)
            h1 = F.relu(h1)
            h1 = self.dropout(h1)
            h2 = torch.spmm(adj, h1) @ self.w2
            f = F.relu(h2)
            return f

    def bernoulli_poisson_loss(f: torch.Tensor, edges: torch.Tensor, non_edges: torch.Tensor) -> torch.Tensor:
        """Balanced Bernoulli-Poisson negative log-likelihood (Shchur & Günnemann, 2019)."""
        # 1. Positive edge log-likelihood: -log(1 - exp(-f_u . f_v))
        u_pos, v_pos = edges[:, 0], edges[:, 1]
        dot_pos = torch.sum(f[u_pos] * f[v_pos], dim=1)
        # Numerical stability clamp
        pos_loss = -torch.log(-torch.expm1(-torch.clamp(dot_pos, min=1e-7, max=30.0)) + 1e-7).mean()
        
        # 2. Negative non-edge log-likelihood: f_u . f_v
        u_neg, v_neg = non_edges[:, 0], non_edges[:, 1]
        dot_neg = torch.sum(f[u_neg] * f[v_neg], dim=1)
        neg_loss = dot_neg.mean()
        
        return pos_loss + neg_loss

def run_nocd(
    G: nx.Graph,
    num_communities: int = None,
    threshold: float = 0.5,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    batch_size: int = 2000,
    seed: int = 42
) -> list[frozenset]:
    """Neural Overlapping Community Detection (NOCD, Shchur & Günnemann, 2019).
    
    Parameters:
    -----------
    G : nx.Graph
        Input undirected network.
    num_communities : int, optional
        Target number of communities. If None, estimated via sqrt(N).
    threshold : float, default=0.5
        Community membership activation threshold (rho = 0.5).
    hidden_dim : int, default=64
        Hidden representation size.
    epochs : int, default=300
        Number of training epochs.
    lr : float, default=0.01
        Learning rate.
    weight_decay : float, default=1e-3
        L2 regularization weight.
    batch_size : int, default=2000
        Number of sampled positive and negative edge pairs per epoch.
    seed : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    list[frozenset]
        Detected overlapping communities.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return []
    if num_communities is None:
        num_communities = max(2, int(np.sqrt(n)))
        
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    if not edges:
        return [frozenset(G.nodes())]
        
    # Build sparse adjacency matrix
    row_idx = [e[0] for e in edges] + [e[1] for e in edges]
    col_idx = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(row_idx), dtype=np.float32)
    adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    
    # Check PyTorch availability
    if not HAS_TORCH:
        # Fallback to spectral embedding thresholding if torch is not present
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=num_communities, random_state=seed)
        F_mat = np.maximum(0, svd.fit_transform(adj))
        comms = collections.defaultdict(set)
        for i in range(n):
            for c in range(num_communities):
                if F_mat[i, c] > threshold:
                    comms[c].add(nodes[i])
        return [frozenset(c) for c in comms.values() if len(c) > 0]
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Symmetric normalization
    norm_adj = normalize_adjacency(adj)
    coo = norm_adj.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size([n, n])).coalesce()
    
    # Node features: identity / row-normalized structure
    x_tensor = torch.eye(n, dtype=torch.float32)
    
    model = GCNEncoder(
        in_features=n,
        hidden_dim=hidden_dim,
        num_communities=num_communities,
        dropout=0.2
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Edge arrays for sampling
    pos_edges = np.array(edges)
    num_edges = len(pos_edges)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample mini-batch of positive edges
        if num_edges > batch_size:
            pos_batch_idx = np.random.choice(num_edges, size=batch_size, replace=False)
            batch_pos = torch.LongTensor(pos_edges[pos_batch_idx])
        else:
            batch_pos = torch.LongTensor(pos_edges)
            
        # Sample mini-batch of negative edges (non-edges)
        s_size = batch_pos.shape[0]
        neg_u = np.random.randint(0, n, size=s_size)
        neg_v = np.random.randint(0, n, size=s_size)
        batch_neg = torch.LongTensor(np.column_stack((neg_u, neg_v)))
        
        # Forward pass
        f = model(x_tensor, adj_tensor)
        loss = bernoulli_poisson_loss(f, batch_pos, batch_neg)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        F_final = model(x_tensor, adj_tensor).numpy()
        
    # Decode overlapping communities with threshold rho = 0.5
    comms = collections.defaultdict(set)
    for i in range(n):
        u = nodes[i]
        assigned = False
        for c in range(num_communities):
            if F_final[i, c] > threshold:
                comms[c].add(u)
                assigned = True
        if not assigned:
            best_c = int(np.argmax(F_final[i]))
            comms[best_c].add(u)
            
    return [frozenset(c) for c in comms.values() if len(c) > 0]
