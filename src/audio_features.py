"""
Audio Feature Extraction and Temporal Segmentation.
Implements:
1. Resampling to 22,050 Hz and track normalization.
2. Log-mel spectrogram extraction (128 bins).
3. Chroma extraction (12 pitch classes).
4. MFCC extraction (20 bins).
5. Fixed-window and beat-synchronous segmentation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf


def load_audio(
    audio_path: str,
    target_sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target_sr, convert to mono, and normalize per track.
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=mono, duration=duration)
    except Exception:
        # Fallback to soundfile if librosa loader encounters file format specifics
        data, sr = sf.read(audio_path)
        if data.ndim > 1 and mono:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        y = data
        if duration is not None:
            max_samples = int(duration * target_sr)
            y = y[:max_samples]

    # Normalize amplitude per track
    max_amp = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if max_amp > 1e-6:
        y = y / max_amp

    return y.astype(np.float32), sr


def extract_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract log-mel spectrogram over T frames (Shape: [n_mels, T]).
    """
    if len(y) == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    # Normalize to [0, 1] or standard scale
    mean_val = np.mean(log_melspec)
    std_val = np.std(log_melspec) + 1e-6
    return ((log_melspec - mean_val) / std_val).astype(np.float32)


def extract_chroma(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_chroma: int = 12
) -> np.ndarray:
    """
    Extract 12-bin Chroma STFT features (Shape: [12, T]).
    """
    if len(y) == 0:
        return np.zeros((n_chroma, 1), dtype=np.float32)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma
    )
    return chroma.astype(np.float32)


def extract_mfcc(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract 20 MFCC features (Shape: [n_mfcc, T]).
    """
    if len(y) == 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.astype(np.float32)


def segment_audio_fixed(
    y: np.ndarray,
    sr: int = 22050,
    segment_duration: float = 5.0,
    segment_hop: float = 2.5
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Split audio into fixed overlapping temporal windows.
    Returns: list of (segment_samples, start_time_sec, end_time_sec)
    """
    total_duration = len(y) / float(sr)
    if total_duration <= 0:
        return []

    window_size = int(segment_duration * sr)
    hop_size = int(segment_hop * sr)

    if len(y) <= window_size:
        # Pad with zeros if shorter than one window
        padded = np.pad(y, (0, window_size - len(y)))
        return [(padded, 0.0, total_duration)]

    segments = []
    for start in range(0, len(y) - window_size + 1, hop_size):
        end = start + window_size
        seg = y[start:end]
        t_start = start / float(sr)
        t_end = end / float(sr)
        segments.append((seg, t_start, t_end))

    # Add remainder if needed
    if len(segments) == 0 or (start + window_size < len(y) and len(y) - start > window_size // 2):
        last_seg = y[-window_size:]
        segments.append((last_seg, (len(y) - window_size) / float(sr), total_duration))

    return segments


def segment_audio_beat(
    y: np.ndarray,
    sr: int = 22050,
    target_segments: int = 10
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment audio synchronously using beat tracking.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        return segment_audio_fixed(y, sr, segment_duration=5.0, segment_hop=2.5)

    # Group beats to form macro-segments of ~target_segments
    stride = max(1, len(beat_times) // target_segments)
    segments = []
    for i in range(0, len(beat_times) - 1, stride):
        t0 = beat_times[i]
        t1 = beat_times[min(i + stride, len(beat_times) - 1)]
        s0 = int(t0 * sr)
        s1 = int(t1 * sr)
        if s1 > s0:
            segments.append((y[s0:s1], float(t0), float(t1)))

    if len(segments) == 0:
        return segment_audio_fixed(y, sr, segment_duration=5.0, segment_hop=2.5)
    return segments


def extract_segment_features(
    y: np.ndarray,
    sr: int = 22050,
    segment_duration: float = 5.0,
    segment_hop: float = 2.5,
    n_mels: int = 128,
    n_chroma: int = 12
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Extract segment-level representations h_i^(0) for graph nodes.
    Each segment is characterized by mean log-mel features (128 dims) + mean chroma (12 dims) = 140 dims.
    Returns:
      node_features: np.ndarray of shape [num_segments, 140]
      timestamps: List of (t_start, t_end) per segment
    """
    segments = segment_audio_fixed(y, sr=sr, segment_duration=segment_duration, segment_hop=segment_hop)
    if len(segments) == 0:
        return np.zeros((1, n_mels + n_chroma), dtype=np.float32), [(0.0, 1.0)]

    features = []
    timestamps = []

    for seg, t_start, t_end in segments:
        melspec = extract_log_mel_spectrogram(seg, sr=sr, n_mels=n_mels)
        chroma = extract_chroma(seg, sr=sr, n_chroma=n_chroma)

        mel_mean = np.mean(melspec, axis=1)    # [128]
        chroma_mean = np.mean(chroma, axis=1)  # [12]

        feat_vector = np.concatenate([mel_mean, chroma_mean], axis=0)
        # Normalize feature vector
        norm = np.linalg.norm(feat_vector) + 1e-8
        features.append(feat_vector / norm)
        timestamps.append((t_start, t_end))

    return np.array(features, dtype=np.float32), timestamps
