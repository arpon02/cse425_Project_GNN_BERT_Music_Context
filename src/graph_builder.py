"""
Music Structure Graph Construction.
Implements:
1. Segment Graphs:
   - Nodes: Time segments with audio feature vectors h_i^(0)
   - Edges: Temporal adjacency + cosine similarity exceeding threshold tau (cos(h_i, h_j) > tau)
2. Chord-Transition Graphs:
   - Nodes: Unique detected chords (triad templates)
   - Edges: Observed transition counts / probabilities
3. Graph Coherence Score calculation:
   - S_graph = (1 / |E|) * sum_{(i,j) in E} I[cos(h_i, h_j) > tau]
4. Serialization to PyTorch Geometric Data (.pt) and JSON (.json).
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
import torch
from torch_geometric.data import Data

# 12 Pitch names
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 24 Standard triad templates (12 Major, 12 Minor)
def _create_chord_templates() -> Dict[str, np.ndarray]:
    templates = {}
    maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)

    for i, name in enumerate(PITCH_NAMES):
        templates[f"{name}"] = np.roll(maj_template, i)
        templates[f"{name}m"] = np.roll(min_template, i)
    return templates

CHORD_TEMPLATES = _create_chord_templates()
CHORD_NAMES = list(CHORD_TEMPLATES.keys())


def build_segment_graph(
    node_features: np.ndarray,
    timestamps: Optional[List[Tuple[float, float]]] = None,
    similarity_threshold: float = 0.7,
    connect_temporal: bool = True,
    add_self_loops: bool = True,
    track_id: str = "sample_track",
    labels: Optional[np.ndarray] = None,
    emotion: Optional[Tuple[float, float]] = None
) -> Data:
    """
    Construct a Segment Graph from segment audio feature representations.
    - Nodes: segments i = 0...N-1
    - Edges:
        * Temporal adjacency: (i, i+1) and (i+1, i)
        * Acoustic similarity: cos(h_i, h_j) > tau
    """
    num_nodes = len(node_features)
    if num_nodes == 0:
        x = torch.zeros((1, 140), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=1)

    # Normalize node features for cosine similarity
    norms = np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
    normed_features = node_features / norms

    # Cosine similarity matrix
    sim_matrix = np.matmul(normed_features, normed_features.T)

    edge_list = []
    edge_weights = []

    # 1. Temporal adjacency
    if connect_temporal and num_nodes > 1:
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_weights.append(float(sim_matrix[i, i + 1]))
            edge_list.append([i + 1, i])
            edge_weights.append(float(sim_matrix[i + 1, i]))

    # 2. Acoustic similarity edges (non-consecutive)
    for i in range(num_nodes):
        for j in range(i + 2, num_nodes):
            sim = float(sim_matrix[i, j])
            if sim > similarity_threshold:
                edge_list.append([i, j])
                edge_weights.append(sim)
                edge_list.append([j, i])
                edge_weights.append(sim)

    # 3. Optional self-loops
    if add_self_loops:
        for i in range(num_nodes):
            edge_list.append([i, i])
            edge_weights.append(1.0)

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    x = torch.tensor(node_features, dtype=torch.float)

    data_dict = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": num_nodes,
        "track_id": track_id
    }

    if timestamps is not None:
        data_dict["timestamps"] = torch.tensor(timestamps, dtype=torch.float)

    if labels is not None:
        data_dict["y"] = torch.tensor(labels, dtype=torch.float)

    if emotion is not None:
        data_dict["emotion"] = torch.tensor(emotion, dtype=torch.float)

    return Data(**data_dict)


def build_chord_transition_graph(
    chroma: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    track_id: str = "sample_track",
    labels: Optional[np.ndarray] = None
) -> Data:
    """
    Construct a Chord Transition Graph.
    - Frames are classified into 24 standard triads via template matching.
    - Nodes: 24 chords with 12-bin template vector as initial node feature.
    - Edges: Directed transitions weighted by observed counts.
    """
    # chroma shape: [12, T]
    T = chroma.shape[1]
    if T < 2:
        x = torch.tensor(list(CHORD_TEMPLATES.values()), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=24, track_id=track_id)

    # Frame-wise chord classification
    template_matrix = np.stack(list(CHORD_TEMPLATES.values()), axis=0) # [24, 12]
    # Normalize templates
    t_norms = np.linalg.norm(template_matrix, axis=1, keepdims=True) + 1e-8
    norm_templates = template_matrix / t_norms

    # Normalize chroma frames
    c_norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
    norm_chroma = chroma / c_norms

    # Cosine similarities: [24, T]
    cos_sims = np.matmul(norm_templates, norm_chroma)
    chord_sequence = np.argmax(cos_sims, axis=0) # [T]

    # Transition count matrix [24, 24]
    transition_counts = np.zeros((24, 24), dtype=np.float32)
    for t in range(T - 1):
        c1 = chord_sequence[t]
        c2 = chord_sequence[t + 1]
        transition_counts[c1, c2] += 1.0

    # Build edges from non-zero transitions
    edge_list = []
    edge_weights = []
    total_transitions = np.sum(transition_counts) + 1e-8

    for i in range(24):
        for j in range(24):
            if transition_counts[i, j] > 0:
                edge_list.append([i, j])
                edge_weights.append(float(transition_counts[i, j] / total_transitions))

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    x = torch.tensor(template_matrix, dtype=torch.float)

    data_dict = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": 24,
        "track_id": track_id
    }
    if labels is not None:
        data_dict["y"] = torch.tensor(labels, dtype=torch.float)

    return Data(**data_dict)


def compute_graph_coherence(data: Data, tau: float = 0.7) -> float:
    """
    Graph coherence score (specification equation page 6):
    S_graph = (1 / |E|) * sum_{(i,j) in E} I[cos(h_i, h_j) > tau]
    Measures whether graph edges align with repeated musical patterns.
    """
    if data.edge_index.numel() == 0 or data.edge_index.shape[1] == 0:
        return 0.0

    x = data.x
    src, dst = data.edge_index[0], data.edge_index[1]
    # Filter out self-loops for coherence score
    mask = src != dst
    if not torch.any(mask):
        return 1.0

    src_nodes = x[src[mask]]
    dst_nodes = x[dst[mask]]

    # Cosine similarity
    cos_sim = torch.cosine_similarity(src_nodes, dst_nodes, dim=1)
    coherent_edges = (cos_sim > tau).float().sum()
    total_edges = float(mask.sum().item())

    return float((coherent_edges / total_edges).item())


def graph_to_json_dict(data: Data) -> Dict[str, Any]:
    """
    Convert PyTorch Geometric Data object to a human-readable and inspectable JSON dictionary.
    """
    num_nodes = data.num_nodes
    edges = []
    if data.edge_index.numel() > 0:
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        weights = data.edge_attr.squeeze().tolist() if data.edge_attr is not None else [1.0] * len(src)
        if isinstance(weights, float):
            weights = [weights]
        for s, d, w in zip(src, dst, weights):
            edges.append({"source": int(s), "target": int(d), "weight": round(float(w), 4)})

    nodes = []
    for i in range(num_nodes):
        node_dict = {"id": i}
        if hasattr(data, "timestamps") and data.timestamps is not None:
            node_dict["start_sec"] = round(float(data.timestamps[i][0]), 2)
            node_dict["end_sec"] = round(float(data.timestamps[i][1]), 2)
        nodes.append(node_dict)

    result = {
        "track_id": getattr(data, "track_id", "unknown"),
        "num_nodes": int(num_nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "coherence_score": round(compute_graph_coherence(data), 4)
    }

    if hasattr(data, "y") and data.y is not None:
        result["labels"] = data.y.tolist()
    if hasattr(data, "emotion") and data.emotion is not None:
        result["emotion"] = data.emotion.tolist()

    return result


def save_graph(data: Data, pt_path: str, json_path: Optional[str] = None):
    """
    Save graph to both .pt and optional .json file.
    """
    torch.save(data, pt_path)
    if json_path is not None:
        dict_rep = graph_to_json_dict(data)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dict_rep, f, indent=2)
