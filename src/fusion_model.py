"""
GNN-BERT Multi-Modal Fusion Model for Multi-Context Understanding (Task 3).
Implements:
1. Cross-Attention Fusion:
   - Query: Q = g * W_Q  (structural graph summary in R^d)
   - Key:   K = H_text * W_K (contextual token embeddings in R^{L x d})
   - Value: V = H_text * W_V
   - Attention: A = softmax(Q K^T / sqrt(d))
   - Context vector: c = A * V
   - Fused representation: z = CONCAT(g, c)
2. Early Concat Fusion:
   - z = CONCAT(g, t_cls)
3. Multi-Task Readout Heads:
   - Multi-label context tagging: y_hat = sigma(W_y * z + b_y)
   - Emotion regression (Valence, Arousal): v_hat, a_hat = W_e * z + b_e
4. Multi-Task Loss:
   L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn_model import MusicGNNEncoder
from src.bert_encoder import MusicBERTEncoder


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism between graph representation g (Query)
    and text token representations H_text (Key, Value).
    """
    def __init__(self, g_dim: int, text_dim: int, attn_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.w_q = nn.Linear(g_dim, attn_dim)
        self.w_k = nn.Linear(text_dim, attn_dim)
        self.w_v = nn.Linear(text_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, attn_dim)

    def forward(
        self,
        g: torch.Tensor,
        h_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: [Batch_size, g_dim] graph readout summary
            h_text: [Batch_size, seq_len, text_dim] text token embeddings
            text_mask: [Batch_size, seq_len] 1 for valid tokens, 0 for pad
        Returns:
            context: [Batch_size, attn_dim] cross-attended text summary
            attn_weights: [Batch_size, seq_len] attention weights over tokens
        """
        B = g.size(0)
        L = h_text.size(1)

        # Q: [B, 1, attn_dim]
        Q = self.w_q(g).unsqueeze(1)
        # K, V: [B, L, attn_dim]
        K = self.w_k(h_text)
        V = self.w_v(h_text)

        # Multi-head reshape
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)     # [B, heads, 1, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)     # [B, heads, L, head_dim]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)     # [B, heads, L, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, heads, 1, L]

        if text_mask is not None:
            mask = text_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # [B, heads, 1, L]
        context = torch.matmul(attn_weights, V)  # [B, heads, 1, head_dim]

        context = context.transpose(1, 2).contiguous().view(B, self.attn_dim)
        context = self.out_proj(context)

        # Average attention weights across heads for visualization [B, L]
        avg_attn = attn_weights.squeeze(2).mean(dim=1)
        return context, avg_attn


class GNNBERTFusionModel(nn.Module):
    """
    Complete Multi-Modal Fusion Model.
    Supports both 'cross_attention' and 'early_concat' fusion.
    """
    def __init__(
        self,
        num_classes: int,
        node_in_dim: int = 140,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        gnn_architecture: str = "GraphSAGE",
        bert_model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = False,
        fusion_mechanism: str = "cross_attention",
        attn_dim: int = 128,
        num_heads: int = 4,
        fc_dim: int = 128,
        dropout: float = 0.2,
        predict_emotion: bool = True
    ):
        super().__init__()
        self.fusion_mechanism = fusion_mechanism
        self.predict_emotion = predict_emotion

        # Encoders
        self.gnn = MusicGNNEncoder(
            in_dim=node_in_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            architecture=gnn_architecture,
            dropout=dropout
        )
        self.bert = MusicBERTEncoder(
            model_name=bert_model_name,
            freeze_backbone=freeze_bert
        )

        bert_dim = self.bert.hidden_dim

        # Fusion layer
        if fusion_mechanism == "cross_attention":
            self.cross_attn = CrossAttentionFusion(
                g_dim=gnn_hidden_dim,
                text_dim=bert_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            )
            fused_dim = gnn_hidden_dim + attn_dim
        else: # early_concat
            self.cross_attn = None
            fused_dim = gnn_hidden_dim + bert_dim

        # Shared projection
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task Heads
        self.tag_head = nn.Linear(fc_dim, num_classes)

        if predict_emotion:
            # 2 regression outputs: valence and arousal
            self.emotion_head = nn.Linear(fc_dim, 2)
        else:
            self.emotion_head = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal pipeline.
        Returns:
            tag_logits: [Batch_size, num_classes]
            emotion_preds: [Batch_size, 2] (optional valence, arousal)
            fused_z: [Batch_size, fc_dim] representation for t-SNE
            attn_weights: [Batch_size, seq_len] (if cross_attention)
        """
        # 1. Structural audio graph readout
        _, g = self.gnn(x, edge_index, batch)

        # 2. Text context representations
        h_text, t_cls, _ = self.bert(input_ids, attention_mask)

        # 3. Multi-modal fusion
        attn_weights = None
        if self.fusion_mechanism == "cross_attention":
            context, attn_weights = self.cross_attn(g, h_text, text_mask=attention_mask)
            z_raw = torch.cat([g, context], dim=-1)
        else:
            z_raw = torch.cat([g, t_cls], dim=-1)

        # 4. Latent representation
        z = self.shared_fc(z_raw)

        # 5. Prediction heads
        tag_logits = self.tag_head(z)

        output = {
            "tag_logits": tag_logits,
            "fused_z": z,
            "g": g,
            "t_cls": t_cls
        }

        if attn_weights is not None:
            output["attn_weights"] = attn_weights

        if self.predict_emotion and self.emotion_head is not None:
            output["emotion_preds"] = self.emotion_head(z)

        return output


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    target_tags: torch.Tensor,
    target_emotion: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task loss:
    L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
    """
    tag_logits = outputs["tag_logits"]
    bce_loss = F.binary_cross_entropy_with_logits(tag_logits, target_tags)

    loss_dict = {"tag_loss": float(bce_loss.item()), "valence_loss": 0.0, "arousal_loss": 0.0}
    total_loss = bce_loss

    if target_emotion is not None and "emotion_preds" in outputs:
        emotion_preds = outputs["emotion_preds"]
        v_pred, a_pred = emotion_preds[:, 0], emotion_preds[:, 1]
        v_true, a_true = target_emotion[:, 0], target_emotion[:, 1]

        v_loss = F.mse_loss(v_pred, v_true)
        a_loss = F.mse_loss(a_pred, a_true)

        total_loss = total_loss + alpha * v_loss + beta * a_loss
        loss_dict["valence_loss"] = float(v_loss.item())
        loss_dict["arousal_loss"] = float(a_loss.item())

    loss_dict["total_loss"] = float(total_loss.item())
    return total_loss, loss_dict
