import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset, DataLoader, random_split
# import torchaudio
from torchvision.transforms.functional import resize
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path='best_cnn_classifier.pth'):
    """Trains the model and saves the best version based on validation accuracy."""
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'batch_losses': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0
        # Unpack image, statics, and label

        # Wrap train_loader with tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for i, (inputs, statics, labels) in enumerate(train_pbar):
            # Move all data tensors to the configured device
            inputs, statics, labels = inputs.to(device), statics.to(device), labels.to(device)

            optimizer.zero_grad()
            # Pass both inputs (image) and statics to the model
            outputs = model(inputs, statics) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            history['batch_losses'].append(loss.item()) # Store batch loss for analysis
            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Optional: Print batch progress
            # if (i + 1) % max(1, len(train_loader) // 5) == 0: # Print ~5 times per epoch
            #      print(f'  Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_train_loss = running_loss_train / total_train # Use total_train for average
        epoch_train_acc = correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        correct_val = 0
        total_val = 0
        running_loss_val = 0.0
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Unpack image, statics, and label
            test_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]', leave=False)
            for inputs, statics, labels in test_pbar:
                # Move all data tensors to the configured device
                inputs, statics, labels = inputs.to(device), statics.to(device), labels.to(device)
                # Pass both inputs (image) and statics to the model
                outputs = model(inputs, statics)

                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_loss_val / total_val # Use total_val for average
        epoch_val_acc = correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f'Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_acc:.4f}')
        
        from dev.utils import plot_training_history # Import the plotting function
        train_path = project_root / 'dev' / 'encoder_scratch' / 'classifier_training'
        plot_training_history(history, train_path)

        # Save best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'  -> New best model saved to {model_save_path} (Val Acc: {best_val_acc:.4f})')

    print('--- Finished Training ---')
    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f}")
    return history


# --- Main Execution Block ---
if __name__ == "__main__":
    import datetime

    # --- Configuration ---
    SR = 64                     # Sample rate (Hz) of the accelerometer data
    SEGMENT_DURATION_SEC = 8   # Duration of each data segment for training (seconds)
    SLIDING_WINDOW_SEC = 2      # Sliding window duration (seconds) for segment extraction 
    IMG_SIZE = 64              # Resize mel spectrogram to this square size (pixels)
    BATCH_SIZE = 512             # Number of samples per batch
    NUM_EPOCHS = 25             # Number of training epochs
    LEARNING_RATE = 0.001       # Optimizer learning rate
    VAL_SPLIT = 0.2             # Fraction of data to use for validation (e.g., 0.2 = 20%)
    NUM_WORKERS = 0             # DataLoader workers (0 often best for Windows/debugging)
    current_time = datetime.datetime.now().strftime("%m%d_%H")
    MODEL_SAVE_PREFIX = f'ctxt{SEGMENT_DURATION_SEC*SR}_sr{SR}'
    MODEL_SAVE = f'cnn_classifier_{current_time}_{MODEL_SAVE_PREFIX}.pth'

    # --- Paths (IMPORTANT: Adjust these paths to your project structure) ---
    try:
        # Assumes script is in a subdirectory (like 'src' or 'scratch')
        # and data is in a 'data' directory at the project root.
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2] # Adjust parent level if needed
        DATA_DIR = project_root / "data" / "PPG_ACC_processed_data"
        # The single pkl containing both accelerometer data and activity labels
        DATA_PATH = DATA_DIR / "data.pkl"
    except NameError:
         # Fallback if __file__ is not defined (e.g., interactive session)
        # Provide the absolute path directly if needed.
        print("Warning: __file__ not defined. Using hardcoded project root.")
        project_root = Path("/c:/Users/liams/Documents/GitHub/E6692 Deep Learning/e6692-2025spring-FinalProject-MUSE-lm3963") # CHANGE THIS
        DATA_DIR = project_root / "data" / "PPG_ACC_processed_data"
        # --- Paths (Adjust if necessary) ---
        try:
            script_path = Path(__file__).resolve()
            project_root = script_path.parents[2] # Assumes script is two levels down from root
            DATA_DIR = project_root / "data" / "PPG_ACC_processed_data"
        except NameError:
            # Fallback for interactive sessions
            print("Warning: __file__ not defined. Using hardcoded project root.")
            # !!! CHANGE THIS PATH IF NEEDED !!!
            project_root = Path("c:/Users/liams/Documents/GitHub/E6692 Deep Learning/e6692-2025spring-FinalProject-MUSE-lm3963")
            DATA_DIR = project_root / "data" / "PPG_ACC_processed_data"
        
        # The single pkl containing both accelerometer data and activity labels
        DATA_PATH = DATA_DIR / "data.pkl"

        # --- File Existence Check ---
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found at: {DATA_DIR}")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

        print(f"Project Root: {project_root}")
        print(f"Data Directory: {DATA_DIR}")
        print(f"Using Data File: {DATA_PATH}")

    import sys
    sys.path.append(str(project_root)) # Add project root to sys.path for imports

    model_save_path = project_root / 'models' / 'encoder' / MODEL_SAVE # Save model in the models directory

    # --- Device Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")


    # ------ Dataset & DataLoaders ------

    # --- Transform Def ---
    # Initialize the transform, telling it which device to use for its operations
    from src.utils.dataprep import ActivityDataset, AccelToRGBMel_librosa 
    accel_to_mel = AccelToRGBMel_librosa(
        sample_rate=SR,
        img_size=IMG_SIZE,
        device=DEVICE # Pass the primary device to the transform
    )

    print("\n--- Loading Dataset ---")
    # Define the column containing activity labels and the static feature columns
    ANNOTATION_COLUMN = 'ActivityDescr' # Adjust if your label column has a different name
    STATIC_FEATURES = ['HeartRate', 'Age', 'Gender', 'Height', 'Weight']

    # --- Dataset Initialization ---
    full_dataset = ActivityDataset(
        data_path=DATA_PATH,
        annotation_col=ANNOTATION_COLUMN,
        statics=STATIC_FEATURES,
        transform=accel_to_mel,
        sample_rate=SR,
        sample_length_s=SEGMENT_DURATION_SEC, # Use fixed segment length based on duration
        sliding_window_s=SLIDING_WINDOW_SEC # Sliding window length
    )

    # Split dataset into training and validation sets
    num_samples = len(full_dataset)

    if num_samples == 0:
        raise ValueError("Dataset is empty after initialization. Check data and annotations.")

    num_val = int(VAL_SPLIT * num_samples)
    num_train = num_samples - num_val
    if num_train <= 0 or num_val <= 0:
        raise ValueError(f"Dataset size ({num_samples}) is too small for the validation split ({VAL_SPLIT}). Need at least ~{int(1/VAL_SPLIT)} samples.")
    # --- Dataset Splitting by Subject ---
    print("\n--- Splitting Dataset by Subject ---")
    all_subject_ids = full_dataset.data['SubjectID'].unique()
    if len(all_subject_ids) < 3:
        raise ValueError(f"Not enough unique subjects ({len(all_subject_ids)}) to select 3 for validation. Need at least 3.")

    # Randomly select 3 subjects for validation
    val_subject_ids = np.random.choice(all_subject_ids, size=3, replace=False)
    train_subject_ids = np.setdiff1d(all_subject_ids, val_subject_ids)

    print(f"Validation Subjects: {val_subject_ids}")
    print(f"Training Subjects: {train_subject_ids}")

    # Get indices for train and validation sets
    train_indices = []
    val_indices = []

    # Assuming full_dataset.data is accessible and contains 'SubjectID'
    # and that the order in full_dataset.data corresponds to the dataset's items
    # If full_dataset.get_subject_id(idx) exists, that would be more robust.
    # For now, let's assume direct access or a helper method is available.

    # If ActivityDataset stores subject information per sample, we can iterate
    # For this example, let's assume full_dataset.data is the primary DataFrame
    # and its index corresponds to the dataset samples.

    # Iterate through each window (item) in the full_dataset
    for i in range(len(full_dataset)): # i is the window index
        # Calculate the target original sample index for this window,
        # mirroring how ActivityDataset.__getitem__ determines the relevant sample
        # for labels and statics.
        start_sample_for_window = i * full_dataset.step_samples
        end_sample_for_window = start_sample_for_window + full_dataset.segment_len_samples
        target_original_sample_idx = end_sample_for_window - 1

        # Ensure target_original_sample_idx is a valid index for full_dataset.data
        # full_dataset.data is a DataFrame with 'SubjectID' for each original sample,
        # and its length is total_samples_available.
        if not (0 <= target_original_sample_idx < len(full_dataset.data)):
            print(f"Warning: Skipping window {i} due to out-of-bounds target_original_sample_idx: {target_original_sample_idx} "
                  f"(max index: {len(full_dataset.data) - 1}). This might indicate an issue with dataset parameters.")
            continue

        # Get the SubjectID for the original sample corresponding to this window's target point.
        # full_dataset.data['SubjectID'] is a pandas Series indexed by original sample index.
        sample_subject_id = full_dataset.data['SubjectID'].iloc[target_original_sample_idx]

        if sample_subject_id in val_subject_ids:
            val_indices.append(i) # Add window index 'i' to val_indices
        else:
            train_indices.append(i) # Add window index 'i' to train_indices

    if not val_indices:
        raise ValueError("Validation set is empty. Check subject ID matching.")
    if not train_indices:
        raise ValueError("Training set is empty. Check subject ID matching.")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"Dataset split: Train={len(train_dataset)} samples, Validation={len(val_dataset)} samples")
    print(f"Number of training subjects: {len(train_subject_ids)}")
    print(f"Number of validation subjects: {len(val_subject_ids)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    print("DataLoaders created.")

    # --- Model, Loss, Optimizer ---
    print("\n--- Initializing Model ---")
    num_classes = full_dataset.num_classes
    num_statics = full_dataset.num_statics
    from models.encoder.encoder import SimpleCNN # TODO adjust for prod
    model = SimpleCNN(num_classes=num_classes, num_statics=num_statics, img_size=IMG_SIZE)
    print(model) # Print model summary
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # incorporate class weights
    from collections import Counter
    class_counts_dict = Counter(full_dataset.annotations['label_encoded'])
    class_counts = [class_counts_dict.get(i, 0) for i in range(full_dataset.num_classes)]

    # Normalize for use in CrossEntropyLoss
    total = sum(class_counts)
    weights = [total / c for c in class_counts]
    norm_weights = torch.tensor(weights) / sum(weights)

    criterion = nn.CrossEntropyLoss(weight=norm_weights.to(DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training ---
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        model_save_path=model_save_path
    )
    
    # --- Save Training History ---
    try:
        history_df = pd.DataFrame(history)
        history_csv_path = project_root / 'training_history_encoder.csv'
        history_df.to_csv(history_csv_path, index=False)
        print(f"\nSaved training history to {history_csv_path}")
    except Exception as e:
        print(f"\nCould not save training history to CSV. Error: {e}")

    try:
        from dev.utils import plot_training_history # Import the plotting function
        train_path = project_root / 'dev' / 'encoder_scratch' / 'classifier_training'
        plot_training_history(history, project_root)
    except ImportError:
        print("\nMatplotlib not found. Skipping history plot.")
    except Exception as e:
        print(f"\nCould not generate plot. Error: {e}")
    
    print("\nScript finished.")