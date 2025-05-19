#!/usr/bin/env python3
"""
script_1_data_preparation.py

This script downloads or loads:
- Movement data: UCI HAR dataset for human activity recognition.
- Music data: FMA 'small' subset with text (genre) labels.

It then preprocesses both into a convenient directory structure:
- 'data/uci_har': raw and extracted UCI HAR.
- 'data/fma_small': raw and extracted FMA files (audio + metadata).
- 'data/processed': preprocessed NumPy files for training.

Note: FMA 'small' subset is ~7GB compressed. Ensure enough disk space.
"""

import os
import requests
import zipfile
import tarfile
import shutil
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. Movement (Accelerometer) Data: UCI HAR
# -----------------------------------------------------------------------------

def download_uci_har_data(destination_dir="data/uci_har"):
    """
    Downloads the UCI HAR Dataset (if not already downloaded).
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_filename = os.path.join(destination_dir, "UCI_HAR_Dataset.zip")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if not os.path.exists(zip_filename):
        print("Downloading UCI HAR Dataset...")
        r = requests.get(url, stream=True)
        with open(zip_filename, 'wb') as f:
            f.write(r.content)
    else:
        print("UCI HAR Dataset zip already exists.")

    # Unzip
    extract_dir = os.path.join(destination_dir, "UCI_HAR_Dataset")
    if not os.path.exists(extract_dir):
        print("Extracting UCI HAR Dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as z:
            z.extractall(destination_dir)

def preprocess_uci_har_data(source_dir="data/uci_har/UCI_HAR_Dataset", output_dir="data/processed"):
    """
    Loads and preprocesses UCI HAR data into a convenient format (e.g., NumPy arrays).
    Real pipeline: read train/test data from the 'Inertial Signals' subfolders or from X_train, Y_train, etc.
    Here we parse them for demonstration and store random data as placeholders.
    """
    if not os.path.exists(source_dir):
        print("UCI HAR raw data not found. Skipping.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Preprocessing UCI HAR data...")

    # -------------------------------------------------------------------------
    # In reality, you would parse the files like:
    # - 'train/Inertial Signals/total_acc_x_train.txt', etc.
    # - 'train/y_train.txt' and 'test/y_test.txt'
    # For demonstration, we'll create random placeholders to show how you'd store them.
    # -------------------------------------------------------------------------
    dummy_activity_data = np.random.randn(1000, 3)  # e.g., 3-axis accelerometer
    dummy_labels = np.random.randint(0, 6, size=(1000,))  # 6 activity classes

    np.save(os.path.join(output_dir, "accelerometer_features.npy"), dummy_activity_data)
    np.save(os.path.join(output_dir, "activity_labels.npy"), dummy_labels)

    print(f"Saved preprocessed accelerometer features and labels to {output_dir}.")


# -----------------------------------------------------------------------------
# 2. Music (Audio + Text Labels) Data: FMA 'small'
# -----------------------------------------------------------------------------

def download_fma_small(destination_dir="data/fma_small"):
    """
    Downloads the FMA 'small' subset (~7GB) + metadata from the official site:
    https://github.com/mdeff/fma

    This subset includes ~8,000 audio files (30s each) across 8 balanced genres.
    We also download metadata for textual labeling (fma_metadata.zip).
    """
    # FMA small subset
    url_fma_small = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    fma_zip_file = os.path.join(destination_dir, "fma_small.zip")

    # FMA metadata
    url_metadata = "https://os.unil.cloud.switch.ch/fma/metadata.zip"
    metadata_zip_file = os.path.join(destination_dir, "metadata.zip")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Download audio subset if not present
    if not os.path.exists(fma_zip_file):
        print("Downloading FMA small subset (this is large, ~7GB)...")
        with requests.get(url_fma_small, stream=True) as r:
            r.raise_for_status()
            with open(fma_zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("FMA small subset already downloaded.")

    # Download metadata if not present
    if not os.path.exists(metadata_zip_file):
        print("Downloading FMA metadata...")
        with requests.get(url_metadata, stream=True) as r:
            r.raise_for_status()
            with open(metadata_zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("FMA metadata already downloaded.")

    # Unzip audio subset
    if not os.path.exists(os.path.join(destination_dir, "fma_small")):
        print("Extracting fma_small audio files...")
        with zipfile.ZipFile(fma_zip_file, 'r') as z:
            z.extractall(destination_dir)

    # Unzip metadata
    metadata_dir = os.path.join(destination_dir, "fma_metadata")
    if not os.path.exists(metadata_dir):
        print("Extracting FMA metadata...")
        with zipfile.ZipFile(metadata_zip_file, 'r') as z:
            z.extractall(destination_dir)

def preprocess_fma_small(source_dir="data/fma_small", output_dir="data/processed"):
    """
    Extracts and preprocesses FMA 'small' data. We'll:
    1. Parse metadata to get text labels (genres, etc.).
    2. (Optional) Convert audio to spectrograms or embeddings for model training.

    This example only shows how to parse metadata (CSV) for text labels,
    then produce random arrays as placeholders for 'spectrograms'.
    Replace or expand with actual audio feature extraction as needed.
    """
    # 1. Check main directories
    fma_audio_dir = os.path.join(source_dir, "fma_small")
    metadata_dir = os.path.join(source_dir, "fma_metadata")
    if not (os.path.exists(fma_audio_dir) and os.path.exists(metadata_dir)):
        print("FMA data not found. Skipping.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Preprocessing FMA 'small' dataset...")

    # 2. Load track metadata (genres, titles, etc.)
    # The main file is 'tracks.csv' in the metadata folder. 
    tracks_csv = os.path.join(metadata_dir, "tracks.csv")
    if not os.path.exists(tracks_csv):
        print("tracks.csv not found in metadata. Skipping.")
        return

    df_tracks = pd.read_csv(tracks_csv, header=[0,1], index_col=0)
    # The CSV has multi-level columns. For instance, 'track' and 'genre' columns at different levels.
    # We'll do a simplified parse. For example, 'track', 'genre', etc. are nested. We can flatten them:
    df_tracks.columns = ['_'.join(col).strip() for col in df_tracks.columns.values]
    # e.g. 'track_genres', 'track_title', 'album_title', etc.

    # 3. Extract some textual label: e.g., the "track_genres" column
    #    which might contain a list of genre IDs. Another approach is the 'track_genre_top'.
    if 'track_genres_all' in df_tracks.columns:
        # The 'track_genres_all' might contain a list of JSON with multiple genres. We'll skip that complexity.
        pass

    # The "track_genre_top" column is simpler: the top genre name.
    if 'track_genre_top' not in df_tracks.columns:
        # fallback
        print("No 'track_genre_top' found. Possibly the dataset format changed.")
    else:
        # This is a single string label, e.g. "Rock", "Pop", "Hip-Hop".
        text_labels = df_tracks['track_genre_top']
        # Drop missing or invalid entries
        text_labels = text_labels.dropna()
        print(f"Sample of genre text labels:\n{text_labels.head()}")

    # You might join audio IDs to the above data. For demonstration, let's assume
    # we store track IDs + top genre string in a CSV to illustrate text labels.
    label_output = os.path.join(output_dir, "fma_genre_labels.csv")
    df_tracks['track_genre_top'].to_csv(label_output, header=True)
    print(f"Saved top-genre labels to {label_output}")

    # 4. (Optional) Audio to Spectrogram conversion.
    #    Real scenario: parse each .mp3 in fma_small/<folder>/<track_id>.mp3
    #    and convert to a log-mel-spectrogram with librosa, then store as .npy or .pt
    # For demonstration, let's generate random spectrogram data with shape (num_tracks, freq_bins, time_frames).
    # We'll pretend we found 2000 tracks with valid audio.
    num_tracks = 2000  # for demonstration
    dummy_spectrograms = np.random.randn(num_tracks, 128, 128).astype(np.float32)
    np.save(os.path.join(output_dir, "fma_spectrograms.npy"), dummy_spectrograms)
    print(f"Saved dummy spectrogram array (shape: {dummy_spectrograms.shape}) to {output_dir}.")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # 1. Movement data (UCI HAR)
    download_uci_har_data()
    preprocess_uci_har_data()

    # 2. Music data (FMA 'small')
    download_fma_small()
    preprocess_fma_small()

    print("Data preparation completed.")

if __name__ == "__main__":
    main()
