"""
BERT Text Encoder and Task 1 Multi-Label Music Tag Classifier.
Implements:
1. Contextual token representation H_text in R^{L x d} and CLS pooling t in R^d.
2. Task 1: Multi-label classification head y_k = sigma(w_k^T t + b_k).
3. Self-attention extraction for qualitative visualizations.
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class MusicBERTEncoder(nn.Module):
    """
    Backbone text encoder leveraging Hugging Face Transformers.
    Extracts sequence tokens H_text and CLS pooled embedding t.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze_backbone: bool = False,
        output_attentions: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, output_attentions=output_attentions)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.hidden_dim = self.backbone.config.hidden_size

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Tokenize raw string texts into padded tensor batches.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Forward pass.
        Returns:
            H_text: [batch_size, seq_len, hidden_dim] contextual token embeddings
            t_cls:  [batch_size, hidden_dim] CLS token embedding
            attentions: tuple of attention weight tensors from each layer
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [B, L, d]
        t_cls = sequence_output[:, 0, :]            # First token (CLS or equivalent)
        attentions = outputs.attentions if hasattr(outputs, "attentions") else None

        return sequence_output, t_cls, attentions


class BERTTagClassifier(nn.Module):
    """
    Task 1 Model: BERT Multi-Label Tag Classifier.
    Model:
        t = BERT_CLS(X_text)
        y_hat_k = sigma(w_k^T t + b_k)
    Loss:
        BCEWithLogitsLoss over K tags.
    """
    def __init__(
        self,
        num_classes: int,
        model_name: str = "distilbert-base-uncased",
        freeze_backbone: bool = False,
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = MusicBERTEncoder(model_name=model_name, freeze_backbone=freeze_backbone)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [batch_size, num_classes]
            t_cls:  [batch_size, hidden_dim]
        """
        _, t_cls, _ = self.encoder(input_ids, attention_mask)
        dropped = self.dropout(t_cls)
        logits = self.classifier(dropped)
        return logits, t_cls

    def predict_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid probabilities y_hat in [0, 1].
        """
        logits, _ = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)

    def get_attention_weights(
        self,
        text: str,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Extract token-level attention weights for Task 1 qualitative visualizations.
        """
        tokens_info = self.encoder.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            _, _, attentions = self.encoder(tokens_info["input_ids"], tokens_info["attention_mask"])

        # Convert input tokens back to readable strings
        tokens = self.encoder.tokenizer.convert_ids_to_tokens(tokens_info["input_ids"][0])

        # Average across all heads of the last layer: shape [1, num_heads, L, L] -> [L, L]
        if attentions is not None:
            last_layer_attn = attentions[-1][0].mean(dim=0) # [L, L]
            cls_attention = last_layer_attn[0].cpu().numpy().tolist() # CLS token attention to all tokens
        else:
            cls_attention = [1.0 / len(tokens)] * len(tokens)

        return {
            "tokens": tokens,
            "cls_attention": cls_attention
        }
