"""
GNN-Based BERT for Understanding Context from Music.
Modules:
- audio_features: Librosa audio feature extraction and segmentation
- graph_builder: Music structure graph construction (segment & chord graphs)
- bert_encoder: Text context representation and Task 1 classifier
- gnn_model: GraphSAGE and GAT encoders for Task 2 and CNN baseline
- fusion_model: Task 3 Cross-attention multi-modal fusion and multi-task heads
- contrastive: Task 4 Dual-encoder cross-modal retrieval with InfoNCE
- baselines: Random, majority-class, and spectrogram CNN baselines
- train: Training workflows for all tasks
- evaluate: Metric calculations, t-SNE, PR curves, retrieval evaluations
"""

__version__ = "1.0.0"
