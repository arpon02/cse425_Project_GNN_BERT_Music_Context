"""
Graph Neural Network Architectures for Music Structure Graphs & CNN Baseline.
Implements:
1. Task 2: GraphSAGE and GAT encoders for segment and chord graphs.
   - Message passing: h_i^(l+1) = sigma(W^(l) * CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)))
   - Readout: g = (1 / |V|) * sum_{i in V} h_i^(L)
   - Multi-label / genre classification head: y_hat = sigma(W g + b)
2. Baseline B2: 2D CNN baseline on log-mel spectrograms.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class NativeSAGEConv(nn.Module):
    """
    Pure PyTorch fallback for GraphSAGE message passing:
    h_i^(l+1) = sigma(W * CONCAT(h_i, MEAN_{j in N(i)} h_j))
    Ensures execution regardless of C++ extensions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            # Self only
            return self.linear(torch.cat([x, x], dim=-1))

        src, dst = edge_index[0], edge_index[1]
        # Aggregate neighbors by mean
        deg = torch.zeros(num_nodes, device=x.device).scatter_add_(
            0, dst, torch.ones_like(dst, dtype=torch.float)
        ).clamp(min=1.0).unsqueeze(-1)

        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src])
        neigh_mean = neigh_sum / deg

        combined = torch.cat([x, neigh_mean], dim=-1)
        return self.linear(combined)


class MusicGNNEncoder(nn.Module):
    """
    Music Graph Neural Network Encoder.
    Outputs both node representations and pooled graph representation g in R^d.
    """
    def __init__(
        self,
        in_dim: int = 140,
        hidden_dim: int = 128,
        num_layers: int = 2,
        architecture: str = "GraphSAGE",
        dropout: float = 0.2
    ):
        super().__init__()
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        current_dim = in_dim

        for i in range(num_layers):
            if HAS_PYG and architecture == "GraphSAGE":
                conv = SAGEConv(current_dim, hidden_dim, aggr="mean")
            elif HAS_PYG and architecture == "GAT":
                conv = GATConv(current_dim, hidden_dim // 4, heads=4, concat=True)
            else:
                conv = NativeSAGEConv(current_dim, hidden_dim)
            self.convs.append(conv)
            current_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            h: Node representations [N, hidden_dim]
            g: Graph-level readout [Batch_size, hidden_dim]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph Readout (Mean pooling over nodes)
        if batch is not None:
            if HAS_PYG:
                g = global_mean_pool(h, batch)
            else:
                num_graphs = int(batch.max().item()) + 1
                g_list = [h[batch == b].mean(dim=0) for b in range(num_graphs)]
                g = torch.stack(g_list, dim=0)
        else:
            # Single graph readout
            g = h.mean(dim=0, keepdim=True)

        return h, g


class MusicGNNClassifier(nn.Module):
    """
    Task 2 Model: GNN on Music Structure Graphs for Genre / Multi-Label Tag Prediction.
    Model:
        h_i^(l+1) = sigma(W^(l) * CONCAT(h_i, MEAN_{j in N(i)} h_j))
        g = MEANPOOL({h_i^(L)})
        y_hat = sigma(W g + b)
    """
    def __init__(
        self,
        in_dim: int = 140,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 2,
        architecture: str = "GraphSAGE",
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = MusicGNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            architecture=architecture,
            dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [Batch_size, num_classes]
            g:      [Batch_size, hidden_dim]
        """
        _, g = self.encoder(x, edge_index, batch)
        logits = self.classifier(g)
        return logits, g

    def predict_probs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits, _ = self.forward(x, edge_index, batch)
        return torch.sigmoid(logits)


class MelSpectrogramCNN(nn.Module):
    """
    Baseline B2: 2D Convolutional Neural Network operating on raw log-mel spectrograms.
    Architecture: Conv2D -> BatchNorm -> ReLU -> MaxPool x3 -> AdaptivePool -> MLP.
    Provides standard baseline for comparing against structure-aware GNN.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [Batch_size, 1, 128, T] or [Batch_size, 128, T]
        Returns:
            logits: [Batch_size, num_classes]
        """
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1) # Add channel dim
        feat = self.features(mel_spec)
        logits = self.classifier(feat)
        return logits
