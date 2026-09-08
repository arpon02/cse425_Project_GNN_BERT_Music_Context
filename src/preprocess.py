"""
Preprocessing Pipeline for Audio Tracks into Music Graphs and Metadata Splits.
Transforms raw audio in data/raw/ into:
1. Segment graphs (.pt and .json in data/processed/graphs/)
2. Log-mel spectrogram caches
3. Train/Val/Test split metadata (data/splits/)
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
from src.dataset import build_splits, DEFAULT_GENRES, DEFAULT_TAGS


def process_audio_file(
    audio_path: str,
    output_dir: str = "data/processed/graphs",
    mel_cache_dir: str = "data/processed/mel",
    similarity_threshold: float = 0.7,
    sample_rate: int = 22050,
    segment_duration: float = 5.0,
    segment_hop: float = 2.5
) -> Dict[str, Any]:
    """
    Process a single audio file into graph and spectrogram representations.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mel_cache_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    track_id = base_name

    # Load and normalize audio
    y, sr = load_audio(audio_path, target_sr=sample_rate)

    # 1. Extract segment features for graph
    node_features, timestamps = extract_segment_features(
        y, sr=sr, segment_duration=segment_duration, segment_hop=segment_hop
    )

    # 2. Build and save segment graph
    graph_data = build_segment_graph(
        node_features=node_features,
        timestamps=timestamps,
        similarity_threshold=similarity_threshold,
        track_id=track_id
    )

    pt_path = os.path.join(output_dir, f"{track_id}.pt")
    json_path = os.path.join(output_dir, f"{track_id}.json")
    save_graph(graph_data, pt_path=pt_path, json_path=json_path)

    # 3. Cache log-mel spectrogram for CNN baseline
    mel_spec = extract_log_mel_spectrogram(y, sr=sr, n_mels=128)
    mel_path = os.path.join(mel_cache_dir, f"{track_id}_mel.npy")
    np.save(mel_path, mel_spec)

    return {
        "track_id": track_id,
        "audio_path": audio_path,
        "mel_path": mel_path,
        "pt_path": pt_path,
        "json_path": json_path,
        "num_nodes": int(graph_data.num_nodes),
        "num_edges": int(graph_data.edge_index.shape[1]) if graph_data.edge_index.numel() > 0 else 0
    }


def scan_and_preprocess(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed/graphs",
    mel_cache_dir: str = "data/processed/mel",
    splits_dir: str = "data/splits",
    similarity_threshold: float = 0.7,
    max_tracks: int = None
):
    """
    Scan data/raw/ directory for all audio files and metadata, preprocess them,
    and generate train/val/test splits.
    """
    audio_extensions = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.au")
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

    audio_files = sorted(audio_files)

    if len(audio_files) == 0:
        print("\n" + "=" * 70)
        print("NO AUDIO FILES FOUND in 'data/raw/'!")
        print("Please place your dataset files in 'data/raw/'.")
        print("Supported datasets:")
        print("  - GTZAN: Place audio files in data/raw/gtzan/")
        print("  - FMA: Place fma_small in data/raw/fma_small/ and metadata in data/raw/fma_metadata/")
        print("  - MusicCaps: Place musiccaps.csv in data/raw/musiccaps/")
        print("=" * 70 + "\n")
        return []

    # Optional track count limit for fast experimentation / testing
    if max_tracks is not None and max_tracks > 0:
        print(f"Limiting preprocessing to first {max_tracks} tracks (out of {len(audio_files)} found).")
        audio_files = audio_files[:max_tracks]
    else:
        print(f"Discovered {len(audio_files)} audio tracks in {raw_dir}. Beginning preprocessing...")

    # 1. Try loading FMA tracks.csv metadata if available
    fma_genres_map = {}
    fma_splits_map = {}
    fma_csv = os.path.join(raw_dir, "fma_metadata", "tracks.csv")
    if not os.path.exists(fma_csv):
        # Also check alternate locations
        fma_candidates = glob.glob(os.path.join(raw_dir, "**", "tracks.csv"), recursive=True)
        if fma_candidates:
            fma_csv = fma_candidates[0]

    if os.path.exists(fma_csv):
        try:
            print(f"Loading FMA metadata from {fma_csv}...")
            tracks_df = pd.read_csv(fma_csv, index_col=0, header=[0, 1], low_memory=False)
            for tid, row in tracks_df.iterrows():
                try:
                    genre = str(row[('track', 'genre_top')]).lower()
                    split = str(row[('set', 'split')]).lower()
                    if genre and genre != 'nan':
                        fma_genres_map[int(tid)] = genre
                    if split and split != 'nan':
                        fma_splits_map[int(tid)] = split
                except Exception:
                    pass
            print(f"Successfully indexed {len(fma_genres_map)} tracks with genre labels from FMA.")
        except Exception as e:
            print(f"Note: Could not parse FMA multi-index tracks.csv: {e}")

    # 2. Try loading MusicCaps captions if available
    musiccaps_captions = []
    for mc_cand in [
        os.path.join(raw_dir, "musiccaps", "musiccaps.csv"),
        os.path.join(raw_dir, "musiccaps", "musiccaps-public.csv")
    ]:
        if os.path.exists(mc_cand):
            try:
                mdf = pd.read_csv(mc_cand)
                if "caption" in mdf.columns:
                    musiccaps_captions = mdf["caption"].dropna().tolist()
                    print(f"Loaded {len(musiccaps_captions)} expert captions from MusicCaps.")
                break
            except Exception:
                pass

    samples = []

    for idx, audio_path in enumerate(tqdm(audio_files, desc="Preprocessing Tracks")):
        try:
            info = process_audio_file(
                audio_path,
                output_dir=output_dir,
                mel_cache_dir=mel_cache_dir,
                similarity_threshold=similarity_threshold
            )

            # Determine genre
            genre = "rock"
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            try:
                int_id = int(base_name)
                if int_id in fma_genres_map:
                    genre = fma_genres_map[int_id]
            except ValueError:
                parent_folder = os.path.basename(os.path.dirname(audio_path)).lower()
                if parent_folder in DEFAULT_GENRES:
                    genre = parent_folder

            # Create multi-label context tag vector
            tags_vector = [1.0 if g.lower() in genre.lower() else 0.0 for g in DEFAULT_GENRES]
            if sum(tags_vector) == 0:
                tags_vector[0] = 1.0  # fallback

            # Select caption (use real MusicCaps caption if available, else descriptive template)
            if musiccaps_captions:
                caption = musiccaps_captions[idx % len(musiccaps_captions)]
            else:
                caption = f"A {genre} music track featuring dynamic rhythms and characteristic instrumentation."

            # Synthetic valence/arousal for DEAM regression (scale 1-9)
            emotion = [5.0, 5.0]

            sample_dict = {
                "track_id": info["track_id"],
                "audio_path": audio_path,
                "mel_path": info["mel_path"],
                "genre": genre,
                "tags": tags_vector,
                "emotion": emotion,
                "caption": caption,
                "text": caption
            }
            samples.append(sample_dict)

        except Exception as e:
            print(f"Warning: Failed to process {audio_path}: {e}")

    print(f"\nSuccessfully preprocessed {len(samples)} tracks.")
    train, val, test = build_splits(samples, splits_dir=splits_dir)
    print(f"Generated splits: {len(train)} train, {len(val)} val, {len(test)} test.")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw audio into music structure graphs.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw audio dataset folder.")
    parser.add_argument("--output_dir", type=str, default="data/processed/graphs", help="Output directory for graphs.")
    parser.add_argument("--mel_dir", type=str, default="data/processed/mel", help="Output directory for mel spectrograms.")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Output directory for split JSONs.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Cosine similarity threshold tau.")
    parser.add_argument("--max_tracks", type=int, default=None, help="Optional maximum number of tracks to process.")
    args = parser.parse_args()

    scan_and_preprocess(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        mel_cache_dir=args.mel_dir,
        splits_dir=args.splits_dir,
        similarity_threshold=args.threshold,
        max_tracks=args.max_tracks
    )
