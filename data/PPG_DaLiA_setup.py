import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()             # …/your_script.py
PROJECT_ROOT = THIS_FILE.parents[1]              # adjust depth as needed
DATA_DIR = PROJECT_ROOT / "data" / "ppg+dalia" / "data" / "PPG_FieldStudy"
PROCESSED_DIR = PROJECT_ROOT / "data" / "PPG_ACC_processed_data"

# make sure the path is import-safe for your own repos/modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ACTIVITY_LABELS = {
    0: "Transition",            # transient / unlabeled
    1: "Sitting and Reading",
    2: "Climbing Stairs",
    3: "Playing Table Soccer",
    4: "Cycling Outdoors",
    5: "Driving a Car",
    6: "Lunch Break",
    7: "Walking",
    8: "Working at Desk",
}

def prep_training_data(
    data_dir: Path = DATA_DIR,
    processed_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    """
    Load the raw PPG-DaLiA files located in *data_dir*, resample /
    interpolate the signals so everything aligns to the PPG timeline,
    attach per-subject metadata, and write a single pkl to
    *processed_dir / 'data.pkl'*.

    Returns
    -------
    pd.DataFrame
        One row per PPG sample, with ACC-xyz, heart-rate label,
        integer activity-ID, text activity description, and subject
        demographics.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"PPG-DaLiA directory not found at {data_dir}. "
            "Download the dataset and/or point DATA_DIR to it."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Accumulators
    # ----------------------------------------------------------------
    ppg_all, acc_all, hr_all, act_all = [], [], [], []
    subj_id_all, age_all, gender_all = [], [], []
    height_all, weight_all, skin_all, sport_all = [], [], [], []

    for subj in range(1, 16):                       # S1 … S15
        pkl_path  = data_dir / f"S{subj}" / f"S{subj}.pkl"
        # Note: The original code had 'save_path' here, but it seems intended
        # to be the path for questionnaire data, let's call it quest_path.
        quest_path  = data_dir / f"S{subj}" / f"S{subj}_quest.csv" # Assuming CSV based on parsing logic

        if not pkl_path.exists():
            print(f"[WARN] {pkl_path} missing – skipping subject {subj}")
            continue

        # --------------------- signals ------------------------------
        with pkl_path.open("rb") as fh:
            raw = pickle.load(fh, encoding="latin1")

        ppg = np.asarray(raw["signal"]["wrist"]["BVP"])      # 64 Hz
        acc = np.asarray(raw["signal"]["wrist"]["ACC"])      # 32 Hz
        hr  = np.asarray(raw["label"])                       #   1 Hz
        act = np.asarray(raw["activity"])                    #   1 Hz

        # --------------------- resample ACC to N Hz ---------------
        acc_interp = np.zeros((ppg.shape[0], acc.shape[1]))
        for i in range(acc.shape[1]):
            acc_interp[:, i] = np.interp(
                np.linspace(0, len(acc) - 1, num=len(ppg)),
                np.arange(len(acc)),
                acc[:, i]
            )
        # --------------------- resample HR & activity --------------
        hr_interp = np.interp(
            np.linspace(0, len(hr) - 1, num=len(ppg)),
            np.arange(len(hr)),
            hr
        )

        act_interp = np.round(np.interp(
            np.linspace(0, len(act) - 1, num=len(ppg)),
            np.arange(len(act)),
            act.flatten()
        )).astype(int)

        # --------------------- metadata ----------------------------
        # defaults in case the CSV is missing / malformed
        meta = {
            "SUBJECT_ID": f"S{subj}",
            "AGE":        -1,
            "GENDER":     "Unknown",
            "HEIGHT":     -1,
            "WEIGHT":     -1,
            "SKIN":       -1,
            "SPORT":      -1,
        }
        try:
            # Assuming the questionnaire file is CSV-like with key,value pairs
            with quest_path.open() as fh:
                meta.update({
                    k.strip(): v.strip()
                    for k, v in
                    (line.lstrip("#").split(",", 1) for line in fh if line.strip() and ',' in line)
                })
        except FileNotFoundError:
            print(f"[WARN] {quest_path} missing – demographics defaulted for S{subj}")
        except Exception as e:
            print(f"[WARN] Error reading {quest_path} for S{subj}: {e} – demographics defaulted")


        # --------------------- stack subject -----------------------
        n = len(ppg)
        ppg_all.append(ppg)
        acc_all.append(acc_interp)
        hr_all.append(hr_interp)
        act_all.append(act_interp)

        subj_id_all.extend([meta["SUBJECT_ID"]]*n)
        # Handle potential non-integer values from metadata reading
        try:
            age_all.extend([int(meta["AGE"])]*n)
        except ValueError:
            age_all.extend([-1]*n)
        gender_all.extend([meta["GENDER"]]*n)
        try:
            height_all.extend([int(meta["HEIGHT"])]*n)
        except ValueError:
            height_all.extend([-1]*n)
        try:
            weight_all.extend([int(meta["WEIGHT"])]*n)
        except ValueError:
            weight_all.extend([-1]*n)
        try:
            skin_all.extend([int(meta["SKIN"])]*n)
        except ValueError:
            skin_all.extend([-1]*n)
        try:
            sport_all.extend([int(meta["SPORT"])]*n)
        except ValueError:
            sport_all.extend([-1]*n)


    # ----------------------------------------------------------------
    # Concatenate across subjects
    # ----------------------------------------------------------------
    ppg_all = np.concatenate(ppg_all, axis=0).flatten() # Flatten PPG to be 1D for DataFrame
    acc_all = np.concatenate(acc_all, axis=0)
    hr_all  = np.concatenate(hr_all,  axis=0)
    act_all = np.concatenate(act_all, axis=0)

    print(f"→ PPG     : {ppg_all.shape}")
    print(f"→ ACC xyz : {acc_all.shape}")
    print(f"→ HR      : {hr_all.shape}")
    print(f"→ Activity: {act_all.shape}")

    # textual activity labels
    act_descr = [_ACTIVITY_LABELS.get(i, "Unknown") for i in act_all] # Use .get for safety

    df = pd.DataFrame({
        "PPG":      ppg_all, # Store flattened PPG directly
        "ACCi":     acc_all[:, 0],
        "ACCj":     acc_all[:, 1],
        "ACCk":     acc_all[:, 2],
        "HeartRate": hr_all,
        "Activity":  act_all,
        "activity_label": act_descr,
        "SubjectID": subj_id_all,
        "Age":       age_all,
        "Gender":    gender_all,
        "Height":    height_all,
        "Weight":    weight_all,
        "SkinType":  skin_all,
        "SportLevel": sport_all,
    })

    # optional split
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # ----------------------------------------------------------------
    # Persist to disk as pkl
    # ----------------------------------------------------------------
    pkl_save_path = processed_dir / "data.pkl"
    df.to_pickle(pkl_save_path)
    print(f"Saved full dataset → {pkl_save_path}")

    return df

def mel_spectrogram(rawdata):
    """
    Compute the Mel spectrogram for the input raw data.
    """
    import librosa

    # Ensure input is float
    rawdata = rawdata.astype(np.float32)

    # Compute the Short-Time Fourier Transform (STFT)
    n_fft = 2048 # FFT window size - consider adjusting based on signal characteristics
    hop_length = 512 # Number of samples between successive frames - adjust based on desired time resolution
    stft_result = librosa.stft(rawdata, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft_result)


    # Compute the Mel spectrogram
    # sr needs to be specified, assuming 32 Hz based on ACC sampling
    sr = 32
    mel_spec = librosa.feature.melspectrogram(S=stft_magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import librosa.display # Import needed for specshow

    processed_data_path = PROCESSED_DIR / "data.pkl"

    if not processed_data_path.exists():
        input(f"Processed data not found at {processed_data_path}. Press Enter to process the data...")
        df = prep_training_data()
    else:
        print(f"Processed data found. Loading from {processed_data_path}...")
        df = pd.read_pickle(processed_data_path)

    print(f"DataFrame loaded with shape: {df.shape}")
    print(df.head())
    print(df.info()) # Check dtypes and non-null counts

    # Ensure columns exist and are numeric before processing
    required_cols = ['ACCi', 'ACCj', 'ACCk']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing one or more required columns: {required_cols}")

    # Convert to numpy arrays, handling potential non-numeric data if necessary
    try:
        acci_data = df['ACCi'].to_numpy(dtype=np.float32)
        accj_data = df['ACCj'].to_numpy(dtype=np.float32)
        acck_data = df['ACCk'].to_numpy(dtype=np.float32)
    except ValueError as e:
        print(f"Error converting ACC columns to numeric numpy arrays: {e}")
        print("Please check the data cleaning and preprocessing steps.")
        sys.exit(1)


    print(f"Performing spectrogram processing on {len(df)} samples...")
    # Check if data length is sufficient for STFT
    n_fft = 2048
    if len(acci_data) < n_fft:
        print(f"[WARN] Data length ({len(acci_data)}) is less than n_fft ({n_fft}). Spectrogram might be empty or invalid.")
        # Handle this case: maybe skip spectrogram, pad data, or adjust n_fft
        log_mel_speci = np.array([]) # Example: create empty array
    else:
        log_mel_speci = mel_spectrogram(acci_data)

    if len(accj_data) < n_fft:
         print(f"[WARN] Data length ({len(accj_data)}) is less than n_fft ({n_fft}). Spectrogram might be empty or invalid.")
         log_mel_specj = np.array([])
    else:
        log_mel_specj = mel_spectrogram(accj_data)

    if len(acck_data) < n_fft:
        print(f"[WARN] Data length ({len(acck_data)}) is less than n_fft ({n_fft}). Spectrogram might be empty or invalid.")
        log_mel_speck = np.array([])
    else:
        log_mel_speck = mel_spectrogram(acck_data)


    # Save each spectrogram to appropriate file only if they are not empty
    if log_mel_speci.size > 0:
        np.save(PROCESSED_DIR / "log_mel_speci.npy", log_mel_speci)
        print(f"Saved log_mel_speci.npy with shape {log_mel_speci.shape}")
    if log_mel_specj.size > 0:
        np.save(PROCESSED_DIR / "log_mel_specj.npy", log_mel_specj)
        print(f"Saved log_mel_specj.npy with shape {log_mel_specj.shape}")
    if log_mel_speck.size > 0:
        np.save(PROCESSED_DIR / "log_mel_speck.npy", log_mel_speck)
        print(f"Saved log_mel_speck.npy with shape {log_mel_speck.shape}")


    # Display the first frame of log_mel_speci if it exists
    if log_mel_speci.size > 0:
        plt.figure(figsize=(10, 4))
        # Use librosa.display.specshow for better visualization
        librosa.display.specshow(log_mel_speci, sr=32, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Mel Spectrogram - ACCi')
        plt.tight_layout() # Adjust layout
    else:
        print("Skipping spectrogram display for ACCi as it's empty.")


    # Calculate the relative time spent on each activity
    if 'ActivityDescr' in df.columns:
        activity_counts = df['ActivityDescr'].value_counts(normalize=True) * 100

        # Create a pie chart
        plt.figure(figsize=(10, 8)) # Adjusted size for better label visibility
        wedges, texts, autotexts = plt.pie(activity_counts, autopct='%1.1f%%', startangle=140)
        plt.title('Relative Time Spent on Each Activity')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # Add a legend instead of labels on wedges if there are many categories
        plt.legend(wedges, activity_counts.index, title="Activities", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout() # Adjust layout
    else:
        print("Skipping activity pie chart as 'ActivityDescr' column is missing.")

    plt.show()
