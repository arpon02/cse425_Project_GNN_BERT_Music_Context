"""
Cross-Modal Audio-Text Alignment with Contrastive Learning (Task 4).
Implements:
1. Dual-Encoder Architecture (GNN for Audio Graphs + BERT for Text Captions).
2. InfoNCE Contrastive Loss:
   L_NCE = -log( exp(sim(g_i, t_i) / tau) / sum_j exp(sim(g_i, t_j) / tau) )
3. Cross-Modal Retrieval Evaluation:
   - Caption -> Audio: R@1, R@5, R@10
   - Audio -> Caption: R@1, R@5, R@10
4. Qualitative retrieval generation (Query Caption -> Top-3 audio clips).
5. Zero-shot tag classification from text prompts.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn_model import MusicGNNEncoder
from src.bert_encoder import MusicBERTEncoder


class ContrastiveDualEncoder(nn.Module):
    """
    Dual-Encoder for joint representation learning between
    music structure graphs (GNN) and natural-language captions (BERT).
    """
    def __init__(
        self,
        node_in_dim: int = 140,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        gnn_architecture: str = "GraphSAGE",
        bert_model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = False,
        projection_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.2
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)

        # 1. Graph Encoder + Projection Head
        self.gnn = MusicGNNEncoder(
            in_dim=node_in_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            architecture=gnn_architecture,
            dropout=dropout
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # 2. Text Encoder + Projection Head
        self.bert = MusicBERTEncoder(
            model_name=bert_model_name,
            freeze_backbone=freeze_bert
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def encode_audio_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract normalized audio graph embedding g_norm in S^{d-1}.
        """
        _, g = self.gnn(x, edge_index, batch)
        proj = self.audio_proj(g)
        return F.normalize(proj, p=2, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract normalized text CLS embedding t_norm in S^{d-1}.
        """
        _, t_cls, _ = self.bert(input_ids, attention_mask)
        proj = self.text_proj(t_cls)
        return F.normalize(proj, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            g_emb: [Batch_size, projection_dim] normalized audio embeddings
            t_emb: [Batch_size, projection_dim] normalized text embeddings
        """
        g_emb = self.encode_audio_graph(x, edge_index, batch)
        t_emb = self.encode_text(input_ids, attention_mask)
        return g_emb, t_emb


def compute_infonce_loss(
    g_emb: torch.Tensor,
    t_emb: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Symmetric InfoNCE Loss (Algorithm 4 / CLIP formulation):
    S_ij = (g_i . t_j) / tau
    L = 0.5 * ( L_audio_to_text + L_text_to_audio )
    """
    B = g_emb.size(0)
    # Cosine similarities matrix [B, B]
    sim_matrix = torch.matmul(g_emb, t_emb.T) / temperature

    labels = torch.arange(B, device=g_emb.device)
    loss_a2t = F.cross_entropy(sim_matrix, labels)
    loss_t2a = F.cross_entropy(sim_matrix.T, labels)

    return 0.5 * (loss_a2t + loss_t2a)


def evaluate_retrieval(
    audio_embeds: np.ndarray,
    text_embeds: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval Recall@K for:
    - Text Caption -> Audio Graph (c2a)
    - Audio Graph -> Text Caption (a2c)
    """
    N = audio_embeds.shape[0]
    if N == 0:
        return {}

    # Similarity matrix: [N, N]
    sims = np.matmul(text_embeds, audio_embeds.T) # sims[i, j] = sim(text_i, audio_j)

    results = {}

    # Caption -> Audio retrieval (each row i retrieves closest audio j)
    # Ground truth is diagonal: j == i
    c2a_ranks = []
    for i in range(N):
        ranked_indices = np.argsort(-sims[i])
        rank = np.where(ranked_indices == i)[0][0] + 1
        c2a_ranks.append(rank)

    for k in k_values:
        r_at_k = np.mean([1.0 if r <= k else 0.0 for r in c2a_ranks])
        results[f"c2a_R@{k}"] = round(float(r_at_k), 4)

    # Audio -> Caption retrieval (each col j retrieves closest text i)
    a2c_ranks = []
    for j in range(N):
        ranked_indices = np.argsort(-sims[:, j])
        rank = np.where(ranked_indices == j)[0][0] + 1
        a2c_ranks.append(rank)

    for k in k_values:
        r_at_k = np.mean([1.0 if r <= k else 0.0 for r in a2c_ranks])
        results[f"a2c_R@{k}"] = round(float(r_at_k), 4)

    results["c2a_MRR"] = round(float(np.mean([1.0 / r for r in c2a_ranks])), 4)
    results["a2c_MRR"] = round(float(np.mean([1.0 / r for r in a2c_ranks])), 4)

    return results


def generate_qualitative_retrieval(
    captions: List[str],
    track_ids: List[str],
    audio_embeds: np.ndarray,
    text_embeds: np.ndarray,
    top_k: int = 3,
    num_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate 10 qualitative retrieval examples (Query Caption -> Top-3 matched clips).
    Satisfies Task 4 deliverable.
    """
    sims = np.matmul(text_embeds, audio_embeds.T)
    N = min(len(captions), num_samples)

    examples = []
    for i in range(N):
        ranked_indices = np.argsort(-sims[i])[:top_k]
        matches = []
        for rank_idx, idx in enumerate(ranked_indices, start=1):
            matches.append({
                "rank": rank_idx,
                "track_id": track_ids[idx],
                "similarity_score": round(float(sims[i, idx]), 4),
                "is_ground_truth": bool(idx == i)
            })

        examples.append({
            "query_index": i,
            "query_caption": captions[i],
            "ground_truth_track": track_ids[i],
            "top_matches": matches
        })

    return examples


def zero_shot_tag_prediction(
    audio_embeds: np.ndarray,
    model: ContrastiveDualEncoder,
    candidate_tags: List[str],
    device: torch.device,
    threshold: float = 0.25
) -> np.ndarray:
    """
    Perform zero-shot multi-label tag prediction by comparing audio graph embeddings
    with text prompt representations e.g. "This music track features {tag}".
    """
    model.eval()
    prompts = [f"This music track features {tag}" for tag in candidate_tags]
    tok = model.bert.tokenize(prompts)
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    with torch.no_grad():
        tag_embeds = model.encode_text(input_ids, attention_mask).cpu().numpy()

    # Audio embeds: [N, D], Tag embeds: [K, D]
    sims = np.matmul(audio_embeds, tag_embeds.T) # [N, K]
    predictions = (sims > threshold).astype(np.float32)
    return predictions
