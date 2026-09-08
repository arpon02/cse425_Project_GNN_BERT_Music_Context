

## . Project Overview 

Music is a multi-layered acoustic signal where semantic **context** spans multiple hierarchies:
- **Harmonic Structure**: Chord sequences, tonal key changes, cadence.
- **Relational Form**: Song structure (verse, chorus, bridge) forming repeated segment graphs.
- **Lyrical Semantics**: Thematic sentiments, poetic narratives, vocal timbre.
- **Descriptive Semantics**: Human listener descriptions, mood tags, genre labels, and continuous emotional valence/arousal.

Traditional sequential audio models (CNNs and RNNs on time-frequency spectrograms) capture local acoustic textures but miss **relational music structure**: how repeated verses link distant song segments, or how chord transitions govern emotional progression.

### Core Goal
This project builds a **hybrid BERT + Graph Neural Network (GNN)** architecture combining:
1. **BERT**: Pretrained contextual language representations ($H_{text} \in \mathbb{R}^{L \times d}$) from lyrics, tags, and natural-language music descriptions (MusicCaps).
2. **GNN**: Message passing on music structure graphs ($G = (\mathcal{V}, \mathcal{E})$) encoding acoustic segment similarities and chord transitions.
3. **Multi-Modal Cross-Attention Fusion**: Dynamically aligns structural audio representations with semantic textual context to solve **multi-label tagging**, **continuous emotion regression (DEAM)**, and **cross-modal audio-text retrieval (MusicCaps)**.





Place  downloaded dataset folders inside `data/raw/`:


## . Repository Structure

```
project1/
├── config.yaml                    # Central hyperparameter & pipeline configuration
├── requirements.txt               # Pinned dependencies (Python 3.10, PyTorch 2.1, PyG)
├── README.md                      # Comprehensive project documentation
├── checkpoints/                   # Saved model weights (.pt)
│   ├── task1_bert.pt
│   ├── task2_gnn.pt
│   ├── baseline_b2_cnn.pt
│   ├── task3_fusion.pt
│   └── task4_contrastive.pt
├── data/
│   ├── raw/                       # Place external datasets here (FMA, GTZAN, etc.)
│   ├── processed/
│   │   ├── graphs/                # 25+ preprocessed .pt and .json graph samples
│   │   └── mel/                   # Cached log-mel spectrograms (.npy)
│   └── splits/                    # train.json, val.json, test.json
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis of audio & graphs
│   └── demo_context.ipynb         # End-to-end interactive inference demo
├── src/
│   ├── __init__.py
│   ├── audio_features.py          # Resampling, 128 log-mel, 12 chroma, segmentation
│   ├── graph_builder.py           # Segment graphs, chord transitions, coherence score
│   ├── bert_encoder.py            # BERT text encoder & Task 1 classifier
│   ├── gnn_model.py               # GraphSAGE / GAT & Baseline B2 Mel-spectrogram CNN
│   ├── fusion_model.py            # Task 3 Cross-Attention Multi-Modal Fusion
│   ├── contrastive.py             # Task 4 InfoNCE Dual-Encoder & Retrieval
│   ├── baselines.py               # Random, Majority-Class, and Handcrafted Baselines
│   ├── dataset.py                 # PyG DataLoaders, dataset wrappers, and split builder
│   ├── preprocess.py              # Scans data/raw/ and generates graphs & splits
│   ├── generate_sample_graphs.py  # Generates reproducible sample graphs
│   ├── train.py                   # Unified CLI trainer for all tasks & ablations
│   └── evaluate.py                # Metrics evaluation, t-SNE, PR curves, case studies
├── results/
│   ├── metrics.json               # Full experimental results table (Table 3)
│   ├── case_studies.md            # 3 detailed qualitative case studies
│   ├── training_histories.json    # Loss & F1 histories per epoch
│   ├── plots/
│   │   ├── tsne_latent_z.png      # 2D t-SNE of fusion space colored by genre
│   │   └── cross_attention_heatmap.png # Audio query over caption tokens
│   └── retrieval_examples/
│       ├── retrieval_results.json # Top-10 qualitative retrieval examples
│       └── retrieval_results.md   # Formatted markdown retrieval table
├── plots/                         # Top-level mirror of visualization plots
├── retrieval_examples/            # Top-level mirror of retrieval deliverables
└── report/                        # Submission report assets
```

---

## . Step-by-Step Execution Guide

### Step 1: Environment Installation
```bash
pip install -r requirements.txt
```
*(Verified compatible with Python 3.10, PyTorch 2.1.2+cu121, CUDA, and PyTorch Geometric).*

### Step 2: Preprocessing Your Data
After placing raw audio in `data/raw/`, run:
```bash
python src/preprocess.py --raw_dir data/raw --threshold 0.7
```

### Step 3: Training the Models
Train all tasks at once or run individual tasks:



