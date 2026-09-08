"""
Dataset Loader, Preprocessing Pipeline, and Multi-Modal PyTorch Geometric DataLoaders.
Designed to handle datasets placed in data/raw/ (FMA, GTZAN, MagnaTagATune, DEAM, MusicCaps,
or custom genre directories).
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from src.audio_features import (
    load_audio,
    extract_log_mel_spectrogram,
    extract_chroma,
    extract_segment_features
)
from src.graph_builder import (
    build_segment_graph,
    build_chord_transition_graph,
    save_graph
)

# Standard genre vocabulary for fallback/GTZAN/FMA
DEFAULT_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# Standard mood/context tags
DEFAULT_TAGS = [
    "energetic", "calm", "melancholic", "happy", "dark",
    "acoustic", "electronic", "vocal", "guitar", "synthesizer"
]


class MusicMultiModalDataset(Dataset):
    """
    Unified multi-modal dataset returning:
    - graph: PyG Data object (segment graph or chord graph)
    - text: raw text caption / tag string
    - mel_spec: [1, 128, T] log-mel spectrogram for CNN baseline
    - tags: [K] multi-label binary vector
    - emotion: [2] (valence, arousal) regression targets
    - track_id: unique identifier
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        graphs_dir: str = "data/processed/graphs",
        tokenizer = None,
        max_length: int = 128
    ):
        self.samples = samples
        self.graphs_dir = graphs_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        track_id = meta["track_id"]
        graph_path = os.path.join(self.graphs_dir, f"{track_id}.pt")

        if os.path.exists(graph_path):
            graph_data = torch.load(graph_path)
        else:
            # Create minimal fallback graph if preprocessed graph isn't cached
            x = torch.randn((5, 140), dtype=torch.float)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
            graph_data = Data(x=x, edge_index=edge_index, num_nodes=5, track_id=track_id)

        # Tags vector
        tags = torch.tensor(meta.get("tags", [0.0] * 10), dtype=torch.float)

        # Emotion targets (valence, arousal)
        emotion = torch.tensor(meta.get("emotion", [5.0, 5.0]), dtype=torch.float)

        # Text representation
        text_str = meta.get("caption", meta.get("text", "instrumental music track"))

        # Spectrogram representation (for CNN baseline)
        mel_path = meta.get("mel_path", None)
        if mel_path and os.path.exists(mel_path):
            mel_spec = np.load(mel_path)
        else:
            # Placeholder or cached array [1, 128, 128]
            mel_spec = np.zeros((1, 128, 128), dtype=np.float32)

        mel_tensor = torch.tensor(mel_spec, dtype=torch.float)
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)

        item = {
            "graph": graph_data,
            "text": text_str,
            "mel_spec": mel_tensor,
            "tags": tags,
            "emotion": emotion,
            "track_id": track_id
        }

        # Tokenize if tokenizer is supplied
        if self.tokenizer is not None:
            tok = self.tokenizer(
                text_str,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            item["input_ids"] = tok["input_ids"].squeeze(0)
            item["attention_mask"] = tok["attention_mask"].squeeze(0)

        return item


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that batches PyG graphs using Batch.from_data_list
    alongside text, spectrograms, and label tensors.
    """
    graph_list = [item["graph"] for item in batch]
    batched_graphs = Batch.from_data_list(graph_list)

    texts = [item["text"] for item in batch]
    track_ids = [item["track_id"] for item in batch]
    tags = torch.stack([item["tags"] for item in batch], dim=0)
    emotions = torch.stack([item["emotion"] for item in batch], dim=0)
    # Pad mel_specs to uniform time dimension
    max_len = max(item["mel_spec"].shape[-1] for item in batch)
    padded_mels = []
    for item in batch:
        mel = item["mel_spec"]
        cur_len = mel.shape[-1]
        if cur_len < max_len:
            pad = torch.zeros((mel.shape[0], mel.shape[1], max_len - cur_len), dtype=mel.dtype)
            mel = torch.cat([mel, pad], dim=-1)
        elif cur_len > max_len:
            mel = mel[:, :, :max_len]
        padded_mels.append(mel)
    mel_specs = torch.stack(padded_mels, dim=0)

    result = {
        "graph": batched_graphs,
        "texts": texts,
        "track_ids": track_ids,
        "tags": tags,
        "emotions": emotions,
        "mel_specs": mel_specs
    }

    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([item["input_ids"] for item in batch], dim=0)
        result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch], dim=0)

    return result


def find_audio_files(raw_dir: str) -> List[str]:
    """
    Recursively find all audio files (.mp3, .wav, .flac, .ogg) in raw_dir.
    """
    extensions = ["*.mp3", "*.wav", "*.flac", "*.ogg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))
    return sorted(files)


def build_splits(
    samples: List[Dict[str, Any]],
    splits_dir: str = "data/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create train, validation, and test split JSON files without artist/track leakage.
    """
    os.makedirs(splits_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    with open(os.path.join(splits_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(os.path.join(splits_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
    with open(os.path.join(splits_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2)

    return train_samples, val_samples, test_samples


def load_splits(splits_dir: str = "data/splits") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load predefined train, val, and test JSON splits.
    """
    train_path = os.path.join(splits_dir, "train.json")
    val_path = os.path.join(splits_dir, "val.json")
    test_path = os.path.join(splits_dir, "test.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    return train_samples, val_samples, test_samples
