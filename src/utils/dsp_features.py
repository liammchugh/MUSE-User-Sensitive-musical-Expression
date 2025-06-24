import numpy as np
import torch
import librosa
from typing import Tuple, Dict, Any


def _ensure_numpy(waveform):
    was_tensor = torch.is_tensor(waveform)
    if was_tensor:
        waveform = waveform.detach().cpu().numpy()
    return waveform, was_tensor


def _wrap_output(result, was_tensor):
    if was_tensor:
        return torch.from_numpy(result)
    return result


def total_energy(waveform: np.ndarray) -> np.ndarray:
    """
    Compute total signal energy per channel.

    Args:
        waveform: np.ndarray or torch.Tensor, shape (C, T) or (T,) for mono
    Returns:
        energy: np.ndarray or torch.Tensor, shape (C,) or scalar for mono
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    energy = np.sum(waveform ** 2, axis=1)
    return _wrap_output(energy, was_tensor)


def spectral_centroid(waveform: np.ndarray, sr: int, 
                      n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """
    Compute spectral centroid for each channel.

    Returns centroid in Hz for each frame: shape (C, N_frames)
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    centroids = []
    for ch in waveform:
        c = librosa.feature.spectral_centroid(
            y=ch, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        centroids.append(c.squeeze())
    result = np.stack(centroids, axis=0)
    return _wrap_output(result, was_tensor)


def zero_crossing_rate(waveform: np.ndarray, 
                       frame_length: int = 2048, hop_length: int = 256) -> np.ndarray:
    """
    Compute zero-crossing rate per channel and per frame.

    Returns array shape (C, N_frames)
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    zcrs = []
    for ch in waveform:
        z = librosa.feature.zero_crossing_rate(
            y=ch, frame_length=frame_length, hop_length=hop_length
        )
        zcrs.append(z.squeeze())
    result = np.stack(zcrs, axis=0)
    return _wrap_output(result, was_tensor)


def mfcc(waveform: np.ndarray, sr: int, 
         n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 256) -> np.ndarray:
    """
    Compute MFCCs for each channel.

    Returns shape (C, n_mfcc, N_frames)
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    mfccs = []
    for ch in waveform:
        m = librosa.feature.mfcc(
            y=ch, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        mfccs.append(m)
    result = np.stack(mfccs, axis=0)
    return _wrap_output(result, was_tensor)


def spectral_contrast(waveform: np.ndarray, sr: int, 
                      n_bands: int = 6, n_fft: int = 2048, hop_length: int = 256) -> np.ndarray:
    """
    Compute spectral contrast per channel.

    Returns shape (C, n_bands+1, N_frames)
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    contrasts = []
    for ch in waveform:
        sc = librosa.feature.spectral_contrast(
            y=ch, sr=sr, n_bands=n_bands,
            n_fft=n_fft, hop_length=hop_length
        )
        contrasts.append(sc)
    result = np.stack(contrasts, axis=0)
    return _wrap_output(result, was_tensor)


def spectral_flatness(waveform: np.ndarray, 
                      n_fft: int = 2048, hop_length: int = 256) -> np.ndarray:
    """
    Compute spectral flatness per channel and per frame.

    Returns array shape (C, N_frames)
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    flats = []
    for ch in waveform:
        f = librosa.feature.spectral_flatness(
            y=ch, n_fft=n_fft, hop_length=hop_length
        )
        flats.append(f.squeeze())
    result = np.stack(flats, axis=0)
    return _wrap_output(result, was_tensor)


def extract_dsp_features(
    waveform: np.ndarray,
    sr: int,
    params: Dict[str, Any] = None
) -> Dict[str, np.ndarray]:
    """
    Extract a suite of DSP features for a given waveform.

    Args:
        waveform: np.ndarray or torch.Tensor, shape (C, T)
        sr: Sampling rate
        params: Optional parameters for feature functions

    Returns:
        Dictionary mapping feature names to arrays.
    """
    waveform, was_tensor = _ensure_numpy(waveform)
    params = params or {}
    feats: Dict[str, Any] = {}
    feats['energy'] = total_energy(waveform)
    feats['centroid'] = spectral_centroid(
        waveform, sr,
        n_fft=params.get('centroid_n_fft', 1024),
        hop_length=params.get('centroid_hop', 256)
    )
    feats['zcr'] = zero_crossing_rate(
        waveform,
        frame_length=params.get('zcr_frame', 2048),
        hop_length=params.get('zcr_hop', 256)
    )
    feats['mfcc'] = mfcc(
        waveform, sr,
        n_mfcc=params.get('n_mfcc', 13),
        n_fft=params.get('mfcc_n_fft', 2048),
        hop_length=params.get('mfcc_hop', 256)
    )
    feats['contrast'] = spectral_contrast(
        waveform, sr,
        n_bands=params.get('n_bands', 6),
        n_fft=params.get('contrast_n_fft', 2048),
        hop_length=params.get('contrast_hop', 256)
    )
    feats['flatness'] = spectral_flatness(
        waveform,
        n_fft=params.get('flat_n_fft', 2048),
        hop_length=params.get('flat_hop', 256)
    )
    if was_tensor:
        # wrap all outputs as tensors
        for k, v in feats.items():
            if not torch.is_tensor(v):
                feats[k] = torch.from_numpy(v)
    return feats
