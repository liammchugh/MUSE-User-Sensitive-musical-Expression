import torch
from torch import Tensor
from pathlib import Path
import pandas as pd
import numpy as np
# from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
import os
import librosa


# if "KMP_DUPLICATE_LIB_OK" in os.environ:
#     del os.environ["KMP_DUPLICATE_LIB_OK"]

class AccelToRGBMel_librosa:
    def __init__(self, sample_rate=64, img_size=64, device="cpu"):
        self.sample_rate = sample_rate
        self.img_size = img_size

    def __call__(self, accel_waveform: np.ndarray) -> torch.Tensor:
        """
        accel_waveform: (3, N) numpy
        Returns: torch.FloatTensor (3, H, W)
        """
        from skimage.transform import resize

        mels = []
        for i in range(3):  # 3-axis
            mel = librosa.feature.melspectrogram(
                y=accel_waveform[i].cpu().numpy(),  # convert to NumPy array
                sr=self.sample_rate,
                n_fft=256,
                hop_length=32,
                n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_img = resize(mel_db, (self.img_size, self.img_size), mode='constant')
            mels.append(mel_img)

        mels = np.stack(mels, axis=0)  # (3, H, W)
        return torch.tensor(mels, dtype=torch.float32)

class AccelToRGBMel:
    """
    Convert 3-axis accelerometer waveform → (3, H, W) log-mel image.
    Suitable for CLIP/BLIP/VLM fine-tuning.
    Correct STFT params for 32 Hz data.
        Log-compress + [0, 1] normalise.
        Square resize to match common VLM backbones.
        Vectorised mel-spectrogram for speed.
        Graceful CPU fallback.
    """
    def __init__(
        self,
        sample_rate: int = 32,                 # ← 32 Hz after resampling
        win_len_sec: float = 4.0,              # window ≈ n_fft / sr
        hop_frac: float = 0.25,                # 75 % overlap
        normalize: bool = True,               # normalise to [0, 1]
        n_mels: int = 64,
        img_size: int = 224,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        import torchaudio
        n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(sample_rate * win_len_sec))))
        hop_length = int(n_fft * hop_frac)
        

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(device)

        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)
        self.img_size = img_size
        self.device = device
        self.normalize = normalize

    @torch.inference_mode()
    def __call__(self, accel: Tensor) -> Tensor:
        """
        accel : (3, T) float tensor on **any** device
        returns: (3, img_size, img_size) float32 in [0,1]
        """
        accel = accel.to(self.device, dtype=torch.float32)

        # Vectorised: (3, T) → (3, n_mels, frames)
        m = self.melspec(accel)
        m = self.to_db(m)                      # log-scale

        # Normalise each channel separately to [0,1]
        if self.normalize:
            m_min, m_max = m.amin(dim=(1,2), keepdim=True), m.amax(dim=(1,2), keepdim=True)
            m = (m - m_min) / (m_max - m_min + 1e-8)

        # Resize spectrogram (H=n_mels, W=frames) to square image
        m = resize(m.unsqueeze(0), size=(self.img_size, self.img_size), antialias=True).squeeze(0)

        return m  # ready for Vision encoder



# --- Dataset ---
from torch.utils.data import Dataset, DataLoader, random_split
class ActivityDataset(Dataset):
    """
    Dataset for loading accelerometer segments and activity labels.

    Assumes:
    1. A single large CSV (`data_csv_path`) contains all raw accelerometer data
       with columns 'ACCi', 'ACCj', 'ACCk'.
    2. An annotation CSV (`annotations_csv_path`) maps segments within the
       raw data to activity labels. It must contain 'start_sample' and
       'activity_label' columns. If `segment_length_samples` is None,
       it must also contain 'end_sample'.
    """
    def __init__(self, data_path, annotation_col, statics, transform=None, sample_rate=64, sample_length_s=8, sliding_window_s=2, debug=False):
        """
        Args:
            data_csv_path (str/Path): Path to the main data CSV (e.g., data.csv).
            annotation_col (str): Column name in the data CSV containing activity labels.
            statics (str): Column name(s) in the data CSV containing static features.
            transform (callable): The AccelToRGBMel transform instance.
            sample_rate (int): Sample rate of the data.
            sample_length (int, optional): If provided, creates fixed-length
                                                    segments starting at 'start_sample'.
                                                    Otherwise, uses 'start_sample' and 'end_sample'
                                                    from annotations.
        """
        self.debug = debug
        self.transform = transform
        self.sample_rate = sample_rate
        self.sample_length_s = sample_length_s
        self.sliding_window_s = sliding_window_s

        # Calculate segment length and step in samples
        self.segment_len_samples = int(self.sample_length_s * self.sample_rate)
        self.step_samples = int(self.sliding_window_s * self.sample_rate)
        if self.step_samples <= 0:
            raise ValueError("Sliding window step must be positive.")

        self.annotation_col = annotation_col
        accel_cols = ['ACCi', 'ACCj', 'ACCk']
        self.statics = statics

        print(f"Loading main data from: {data_path}")
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            df_data = pd.read_csv(data_path)
            print(f"Loaded data from CSV: {data_path}")
        elif data_path.suffix == '.pkl':
            df_data = pd.read_pickle(data_path)
            print(f"Loaded data from Pickle: {data_path}")
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}. Please provide a .csv or .pkl file.")

        if not all(col in df_data.columns for col in accel_cols):
             raise ValueError(f"Data CSV must contain columns: {accel_cols}. Found: {df_data.columns.tolist()}")
        
        # Pre-convert to tensor: (3, Total_Samples)
        self.accel_data_full = torch.tensor(df_data[accel_cols].values, dtype=torch.float32).T
        self.total_samples_available = self.accel_data_full.shape[1]
        print(f"Loaded {self.total_samples_available} samples.")

        # --- Process Annotations ---
        # Encode labels directly in the main DataFrame to avoid SettingWithCopyWarning
        self.labels = df_data[self.annotation_col].astype('category')
        self.label_map = dict(enumerate(self.labels.cat.categories))
        self.label_map_rev = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        print(f"Found {self.num_classes} classes: {self.label_map}")

        # Add encoded labels as a new column to the original DataFrame
        df_data['label_encoded'] = self.labels.cat.codes

        # Assign the relevant annotation columns (original and encoded) to self.annotations
        # Now self.annotations is a DataFrame, not just a Series slice
        self.annotations = df_data[[self.annotation_col, 'label_encoded']].copy()

        # Extract static features
        if self.statics:
            if not all(col in df_data.columns for col in self.statics):
                raise ValueError(f"Static feature columns {self.statics} not found in data CSV. Found: {df_data.columns.tolist()}")

            # Handle potential non-numeric columns like 'Gender'
            df_statics = df_data[self.statics].copy() # Work on a copy
            for col in self.statics:
                if df_statics[col].dtype == 'object':
                    print(f"Converting object column '{col}' to numeric codes.")
                    df_statics[col] = pd.factorize(df_statics[col])[0] # Use factorize for simple encoding

            # Convert static features to tensor: (Num_Samples, Num_Static_Features)
            self.static_features_full = torch.tensor(df_statics.values, dtype=torch.float32)
            self.num_statics = self.static_features_full.shape[1]
        del df_data # Free memory

        # Calculate the number of possible windows
        if self.total_samples_available < self.segment_len_samples:
            self._num_windows = 0
            print(f"Warning: Total samples ({self.total_samples_available}) is less than segment length ({self.segment_len_samples}). Dataset will be empty.")
        else:
            # The number of windows is 1 + floor((total_length - segment_length) / step_length)
            self._num_windows = 1 + (self.total_samples_available - self.segment_len_samples) // self.step_samples
        print(f"Calculated {self._num_windows} possible windows.")


    def __len__(self):
        """Returns the total number of sliding windows."""
        return self._num_windows

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < 0: # Check lower bound
             raise IndexError(f"Index {idx} cannot be negative.")
        if idx >= self._num_windows:
            raise IndexError(f"Window index {idx} out of bounds for {self._num_windows} windows")

        # Calculate start and end samples based on window index and step/length in samples
        start_sample = idx * self.step_samples
        end_sample = start_sample + self.segment_len_samples

        # --- Get Label and Statics ---
        # Use the sample corresponding to the *end* of the window as the target
        # for label and static features.
        target_sample_idx = end_sample - 1

        # Ensure target index is valid before accessing annotations or statics
        if target_sample_idx < 0 or target_sample_idx >= self.total_samples_available:
             raise IndexError(f"Target sample index {target_sample_idx} is out of bounds "
                      f"[0, {self.total_samples_available}) for window index {idx}.")

        # Get label for the target sample
        # self.annotations is a DataFrame/Series w/'label_encoded' column, indexed from 0 to total_samples_available - 1.
        try:
            # Use .iloc for integer-location based indexing
            label_encoded = self.annotations.iloc[target_sample_idx]['label_encoded']
        except IndexError:
             raise IndexError(f"Cannot access label at index {target_sample_idx} from annotations "
                      f"(length {len(self.annotations)}). Window index {idx}.")
        except KeyError:
             raise KeyError(f"'label_encoded' column not found in self.annotations at index {target_sample_idx}.")

        label = torch.tensor(label_encoded, dtype=torch.long)

        # Get static features for the target sample
        if self.static_features_full is not None:
            # Bounds already checked by target_sample_idx check above
            statics = self.static_features_full[target_sample_idx] # Shape: (Num_Static_Features,)
        else:
            # If no static features were loaded, create an empty tensor
            statics = torch.empty(0, dtype=torch.float32)

        # --- Extract Segment Data ---
        # Ensure calculated indices for the segment are valid
        if start_sample < 0 or end_sample > self.total_samples_available:
             # This check is a safeguard against potential miscalculation or edge cases.
             raise IndexError(f"Calculated segment sample range [{start_sample}, {end_sample}) "
                      f"is out of bounds [0, {self.total_samples_available}) for window index {idx}.")

        segment_data = self.accel_data_full[:, start_sample:end_sample]

        # Verify segment length (important safeguard, especially if transform expects fixed size)
        if segment_data.shape[1] != self.segment_len_samples:
            raise ValueError(f"Extracted segment for window index {idx} has incorrect length: "
                     f"{segment_data.shape[1]}, expected {self.segment_len_samples}. "
                     f"Range was [{start_sample}, {end_sample}).")

        # Apply transform -> (3, H, W) on the transform's device
        image = self.transform(segment_data)

        if self.debug:
            try:
                import matplotlib.pyplot as plt
                # Prepare image: move to CPU, permute to HWC, convert to numpy
                img_display = image.cpu().permute(1, 2, 0).numpy()

                # Normalize image to [0, 1] for display if it's not already
                min_val, max_val = img_display.min(), img_display.max()
                if max_val > min_val:
                    img_display = (img_display - min_val) / (max_val - min_val)
                img_display = np.clip(img_display, 0, 1) # Ensure values are in [0, 1]

                # Get label string
                label_str = self.label_map.get(label.item(), "Unknown")
                # Format statics
                statics_list = [f"{s:.2f}" for s in statics.cpu().numpy()]
                statics_str = f"[{', '.join(statics_list)}]" if statics_list else "None"

                # Plot
                plt.figure(figsize=(6, 7))
                plt.imshow(img_display)
                plt.title(f"Index: {idx}\nLabel: {label_str}\nStatics: {statics_str}")
                plt.axis('off')
                plt.show() # Display the plot immediately

            except ImportError:
                print("Debug plotting skipped: Matplotlib not installed.")
            except Exception as e:
                print(f"Debug plotting failed for index {idx}: {e}")

        # Return image and label on CPU (standard practice for DataLoader)
        return image.cpu(), statics.cpu(), label.cpu()


# --- Load data from PKL ---
def loaddata(path):    
    """Load accelerometer data from pkl file.
    Args:
        path (str): Path to the file.
        sr (int): Sampling rate in Hz.
        sl (int): Length of the signal in seconds.
        start_time (float): Start time in seconds.
    Returns:    
        np.ndarray: Accelerometer data as a numpy array.
    """
    # Define file path - adjust if your structure differs
    try:
        # Assumes script is in src/edge_prcss and data is in data/
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2] # Go up from src/edge_prcss to project root
        data_file = project_root / "data" / "PPG_ACC_processed_data" / "data.pkl"
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive session)
        # Try common relative paths or the absolute path from the user context
        potential_paths = [
            Path("data/PPG_ACC_processed_data/data.pkl"),
            Path("../data/PPG_ACC_processed_data/data.pkl"),
            Path("../../data/PPG_ACC_processed_data/data.pkl"),
            Path("/c:/Users/liams/Documents/GitHub/E6692 Deep Learning/e6692-2025spring-FinalProject-MUSE-lm3963/data/PPG_ACC_processed_data/data.pkl")
        ]
        data_file = None
        for p in potential_paths:
            if p.exists():
                data_file = p.resolve()
                break
        if data_file is None:
             raise FileNotFoundError(f"Data file 'data.pkl' not found in expected locations: {potential_paths}")

    print(f"Loading data from: {data_file}")
    df = pd.read_pickle(data_file)
    return df


if __name__ == "__main__":
    sr = 32                     # after your signal-prep stage
    sl = 30               # 30 seconds of data
    start_time = 1500.0           # start time in seconds

    # Convert to PyTorch tensor (float32) and transpose to shape (3, n_samples)
    dummy_grab = False

    if dummy_grab:
        dummy_data = np.random.rand(3, sr * sl).astype(np.float32)  # Dummy data for testing
        dummy = torch.tensor(dummy_data, dtype=torch.float32).T
    else:
        path = None
        df = loaddata(path, sr, sl, start_time)  
        
    # Identify accelerometer columns (adjust if names differ in your CSV)
    accel_cols = ['ACCi', 'ACCj', 'ACCk']
    if not all(col in df.columns for col in accel_cols):
        # Basic check if standard names aren't present
        print(f"Warning: Columns {accel_cols} not found. Available: {df.columns.tolist()}. Trying to infer.")
        # Add more sophisticated column finding logic here if needed
        # For now, raise error if standard columns aren't found.
        raise ValueError(f"CSV must contain columns: {accel_cols}. Found: {df.columns.tolist()}")

    # Calculate total samples and samples needed
    total_samples = len(df)
    n_samples = sr * sl

    # Ensure enough data is available
    if total_samples < n_samples:
        raise ValueError(
            f"Not enough samples in CSV ({total_samples}) to extract {sl} seconds "
            f"({n_samples} samples at {sr} Hz)."
        )

    # Choose a random start index
    max_start_idx = total_samples - n_samples
    # Use Python's min for comparing two scalars
    start_idx = min(max_start_idx, int(start_time * sr))  # Ensure start time is within bounds
    print(f"Extracting {n_samples} samples starting from index {start_idx}")

    # Slice the dataframe, select columns, convert to numpy array
    accel_data = df.iloc[start_idx : start_idx + n_samples][accel_cols].values
  
    accel_data = torch.tensor(accel_data, dtype=torch.float32).T
    # --- End loading data ---

    xfm = AccelToRGBMel(sample_rate=sr)
    rgb_img = xfm(accel_data)        # (3, 224, 224), float32, [0..1]

    # quick sanity-plot
    import matplotlib.pyplot as plt
    duration_sec = accel_data.shape[1] / sr          # ~30 s
    nyquist      = sr / 2.0                          # 16 Hz
    plt.imshow(
        rgb_img[0].cpu(),
        extent=(0, duration_sec, 0, nyquist),        # Hz scale
        origin="lower", aspect="auto"
    )
    plt.title("Axis‑0 log‑mel (normalised)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # debug only
    import psutil, re, os
    import pandas as pd
    import numpy as np
    from pathlib import Path
    def list_omp_libs():
        proc = psutil.Process()
        return sorted({m.path for m in proc.memory_maps()
                    if re.search(r"(?:lib|i)?omp|gomp", os.path.basename(m.path), re.I)})
    # print("Loaded OpenMP libraries →", list_omp_libs())

    # debug only
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.show()

