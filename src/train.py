"""
Unified Training Pipeline for GNN-BERT Music Context Project.
Supports:
- Task 1: BERT Multi-Label Tag Classifier
- Task 2: GNN on Music Structure Graphs (GraphSAGE / GAT)
- Baseline B2: 2D CNN on Log-Mel Spectrograms
- Task 3: GNN-BERT Fusion (Cross-Attention & Early-Concat) with Multi-Task Loss
- Task 4: Contrastive GNN-BERT InfoNCE Dual-Encoder (MusicCaps)
- Ablation Study: BERT-only, GNN-only, Early Concat, Cross-Attention
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import (
    MusicMultiModalDataset,
    collate_multimodal_batch,
    load_splits
)
from src.bert_encoder import MusicBERTEncoder, BERTTagClassifier
from src.gnn_model import MusicGNNClassifier, MelSpectrogramCNN
from src.fusion_model import GNNBERTFusionModel, compute_multitask_loss
from src.contrastive import ContrastiveDualEncoder, compute_infonce_loss


def train_task1_bert(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 10,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 5,
    lr: float = 2e-5,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/task1_bert.pt"
) -> Dict[str, Any]:
    """Train Task 1 BERT multi-label classifier."""
    print("\n--- Training Task 1: BERT Multi-Label Classifier ---")
    model = BERTTagClassifier(num_classes=num_classes, model_name=model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Task 1 Epoch {epoch}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["tags"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["tags"].to(device)
                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append((probs >= 0.5).astype(np.float32))
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, num_classes))
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_classes))

        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
        micro_f1 = float(f1_score(targets_arr, preds_arr, average="micro", zero_division=0))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(macro_f1)
        history["val_micro_f1"].append(micro_f1)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Macro-F1 = {macro_f1:.4f} | Micro-F1 = {micro_f1:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved Task 1 model checkpoint to {save_path}")
    return {"history": history, "model": model}


def train_task2_gnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_dim: int = 140,
    hidden_dim: int = 128,
    num_classes: int = 10,
    architecture: str = "GraphSAGE",
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/task2_gnn.pt"
) -> Dict[str, Any]:
    """Train Task 2 GNN classifier on segment graphs."""
    print(f"\n--- Training Task 2: GNN ({architecture}) on Music Structure Graphs ---")
    model = MusicGNNClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        architecture=architecture
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Task 2 Epoch {epoch}/{epochs}"):
            graph = batch["graph"].to(device)
            targets = batch["tags"].to(device)

            optimizer.zero_grad()
            logits, _ = model(graph.x, graph.edge_index, getattr(graph, "batch", None))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                graph = batch["graph"].to(device)
                targets = batch["tags"].to(device)
                logits, _ = model(graph.x, graph.edge_index, getattr(graph, "batch", None))
                val_loss += criterion(logits, targets).item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append((probs >= 0.5).astype(np.float32))
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        from sklearn.metrics import f1_score
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, num_classes))
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_classes))
        macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(macro_f1)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Macro-F1 = {macro_f1:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return {"history": history, "model": model}


def train_baseline_b2_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 10,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/baseline_b2_cnn.pt"
) -> Dict[str, Any]:
    """Train Baseline B2: 2D CNN on mel-spectrograms."""
    print("\n--- Training Baseline B2: 2D CNN on Mel-Spectrograms ---")
    model = MelSpectrogramCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"CNN Baseline Epoch {epoch}/{epochs}"):
            mel_specs = batch["mel_specs"].to(device)
            targets = batch["tags"].to(device)

            optimizer.zero_grad()
            logits = model(mel_specs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                mel_specs = batch["mel_specs"].to(device)
                targets = batch["tags"].to(device)
                logits = model(mel_specs)
                val_loss += criterion(logits, targets).item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append((probs >= 0.5).astype(np.float32))
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        from sklearn.metrics import f1_score
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, num_classes))
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_classes))
        macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(macro_f1)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Macro-F1 = {macro_f1:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return {"history": history, "model": model}


def train_task3_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 10,
    fusion_mechanism: str = "cross_attention",
    epochs: int = 5,
    lr: float = 5e-5,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/task3_fusion.pt"
) -> Dict[str, Any]:
    """Train Task 3 GNN-BERT multi-modal fusion model with multi-task loss."""
    print(f"\n--- Training Task 3: GNN-BERT Fusion ({fusion_mechanism}) ---")
    model = GNNBERTFusionModel(
        num_classes=num_classes,
        fusion_mechanism=fusion_mechanism,
        predict_emotion=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_mae_emotion": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Task 3 Epoch {epoch}/{epochs}"):
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags"].to(device)
            emotions = batch["emotions"].to(device)

            optimizer.zero_grad()
            outputs = model(
                x=graph.x,
                edge_index=graph.edge_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch=getattr(graph, "batch", None)
            )
            loss, _ = compute_multitask_loss(outputs, tags, emotions, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        all_v_preds, all_v_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                graph = batch["graph"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tags = batch["tags"].to(device)
                emotions = batch["emotions"].to(device)

                outputs = model(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch=getattr(graph, "batch", None)
                )
                loss, _ = compute_multitask_loss(outputs, tags, emotions, alpha=alpha, beta=beta)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs["tag_logits"]).cpu().numpy()
                all_preds.append((probs >= 0.5).astype(np.float32))
                all_targets.append(tags.cpu().numpy())

                if "emotion_preds" in outputs:
                    all_v_preds.append(outputs["emotion_preds"].cpu().numpy())
                    all_v_trues.append(emotions.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        from sklearn.metrics import f1_score
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, num_classes))
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_classes))
        macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))

        mae = 0.0
        if all_v_preds:
            v_p = np.concatenate(all_v_preds, axis=0)
            v_t = np.concatenate(all_v_trues, axis=0)
            mae = float(np.mean(np.abs(v_p - v_t)))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(macro_f1)
        history["val_mae_emotion"].append(mae)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Macro-F1 = {macro_f1:.4f} | Emotion MAE = {mae:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return {"history": history, "model": model}


def train_task4_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 5e-5,
    temperature: float = 0.07,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/task4_contrastive.pt"
) -> Dict[str, Any]:
    """Train Task 4 InfoNCE Dual-Encoder on paired audio graphs and text captions."""
    print("\n--- Training Task 4: Contrastive Dual-Encoder (InfoNCE) ---")
    model = ContrastiveDualEncoder(temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Task 4 Epoch {epoch}/{epochs}"):
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            g_emb, t_emb = model(
                x=graph.x,
                edge_index=graph.edge_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch=getattr(graph, "batch", None)
            )
            loss = compute_infonce_loss(g_emb, t_emb, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                graph = batch["graph"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                g_emb, t_emb = model(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch=getattr(graph, "batch", None)
                )
                loss = compute_infonce_loss(g_emb, t_emb, temperature=temperature)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch}: Train NCE Loss = {avg_train_loss:.4f} | Val NCE Loss = {avg_val_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return {"history": history, "model": model}


def main():
    parser = argparse.ArgumentParser(description="GNN-BERT Music Context Training Pipeline.")
    parser.add_argument("--task", type=str, default="all", choices=["1", "2", "baseline_b2", "3", "4", "ablation", "all"],
                        help="Task to train.")
    parser.add_argument("--fusion", type=str, default="cross_attention", choices=["cross_attention", "early_concat"],
                        help="Fusion type for Task 3.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per task.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto.")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using compute device: {device}")

    # Load splits
    train_samples, val_samples, test_samples = load_splits("data/splits")
    print(f"Loaded datasets: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples.")

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = MusicMultiModalDataset(train_samples, tokenizer=tokenizer)
    val_ds = MusicMultiModalDataset(val_samples, tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    histories = {}

    if args.task in ["1", "all"]:
        res1 = train_task1_bert(train_loader, val_loader, epochs=args.epochs, device=device)
        histories["task1"] = res1["history"]

    if args.task in ["2", "all"]:
        res2 = train_task2_gnn(train_loader, val_loader, epochs=args.epochs, device=device)
        histories["task2"] = res2["history"]

    if args.task in ["baseline_b2", "all"]:
        res_b2 = train_baseline_b2_cnn(train_loader, val_loader, epochs=args.epochs, device=device)
        histories["baseline_b2"] = res_b2["history"]

    if args.task in ["3", "all"]:
        res3 = train_task3_fusion(train_loader, val_loader, fusion_mechanism=args.fusion, epochs=args.epochs, device=device)
        histories["task3"] = res3["history"]

    if args.task in ["ablation"]:
        # Run both early concat and cross attention
        print("\n--- Running Ablation Study ---")
        train_task3_fusion(train_loader, val_loader, fusion_mechanism="early_concat", epochs=args.epochs, device=device, save_path="checkpoints/ablation_early_concat.pt")
        train_task3_fusion(train_loader, val_loader, fusion_mechanism="cross_attention", epochs=args.epochs, device=device, save_path="checkpoints/ablation_cross_attention.pt")

    if args.task in ["4", "all"]:
        res4 = train_task4_contrastive(train_loader, val_loader, epochs=args.epochs, device=device)
        histories["task4"] = res4["history"]

    # Save histories for evaluate / plotting
    os.makedirs("results", exist_ok=True)
    with open("results/training_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    print("\nTraining complete! Checkpoints stored in 'checkpoints/' and histories in 'results/training_histories.json'.")


if __name__ == "__main__":
    main()
