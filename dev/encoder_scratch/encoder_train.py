import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from torchvision.transforms.functional import resize
from pathlib import Path
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def sigmoid_contrastive_loss(image_embeds, text_embeds, temperature=1.0):
    """
    Implements SigLIP-style binary contrastive loss.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = torch.matmul(image_embeds, text_embeds.T) / temperature
    labels = torch.eye(logits.shape[0]).to(logits.device)  # Identity = correct pairs

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def train_encoder(
    model,
    processor,
    activity_labels,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    muse_tokenizer,
    muse_text_encoder,
    model_save_path='best_siglip_encoder.pth'
):
    model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'batch_losses': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for images, _, labels in train_pbar:
            label_texts = [activity_labels[int(label)] for label in labels]
            optimizer.zero_grad()

            # Process image using SigLIP vision model
            processed = processor(images=images, return_tensors="pt").to(device)
            image_embeds = model.get_image_features(**processed)

            # Get text embeddings from frozen text encoder (ensure alignment with MusicGen encoder)
            text_inputs = muse_tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_out = muse_text_encoder(**text_inputs)
                text_embeds = text_out.last_hidden_state[:, 0, :].to(image_embeds.dtype) # cls token

            loss = sigmoid_contrastive_loss(image_embeds, text_embeds)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            history['batch_losses'].append(loss.item()) # Store batch loss for analysis

        avg_train_loss = running_loss_train / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for images, _, labels in val_pbar:
                label_texts = [activity_labels[int(label)] for label in labels]

                processed = processor(images=images, return_tensors="pt").to(device)
                image_embeds = model.get_image_features(**processed)

                text_inputs = muse_tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_out = muse_text_encoder(**text_inputs)
                text_embeds = text_out.last_hidden_state[:, 0, :].to(image_embeds.dtype) # cls token

                loss = sigmoid_contrastive_loss(image_embeds, text_embeds)
                running_loss_val += loss.item()
    
        avg_val_loss = running_loss_val / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

        
        from dev.utils import plot_training_history # Import the plotting function
        train_path = project_root / 'dev' / 'encoder_scratch' / 'siglip_training'
        plot_training_history(history, train_path)
        
    print('--- Finished Training ---')
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")

    return history



# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    SR = 64                     # Sample rate (Hz) of the accelerometer data
    SEGMENT_DURATION_SEC = 8   # Duration of each data segment for training (seconds)
    SLIDING_WINDOW_SEC = 2      # Sliding window duration (seconds) for segment extraction 
    IMG_SIZE = 64              # Resize mel spectrogram to this square size (pixels)
    BATCH_SIZE = 64             # Number of samples per batch
    NUM_EPOCHS = 3             # Number of training epochs
    LEARNING_RATE = 0.00003       # Optimizer learning rate
    VAL_SPLIT = 0.2             # Fraction of data to use for validation (e.g., 0.2 = 20%)
    NUM_WORKERS = 0             # DataLoader workers (0 often best for Windows/debugging)
    MODEL_SAVE = f'SigLIP2_seglen{SEGMENT_DURATION_SEC}.pth' # Path to save the best model

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

    # --- Importing Modules ---
    import sys
    sys.path.append(str(project_root)) # Add project root to sys.path for imports

    model_save_path = project_root / 'models' / 'encoder' / MODEL_SAVE # Save model in the models directory

    # --- Device Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    from src.utils.dataprep import ActivityDataset, AccelToRGBMel 
    from transformers import AutoProcessor, AutoModel
    import torch.optim as optim

    # --- Transform Definition ---
    accel_to_mel = AccelToRGBMel(
        sample_rate=SR,
        img_size=IMG_SIZE,
        device=DEVICE
    )

    activity_labels = {
        0: 'Climbing Stairs',
        1: 'Cycling Outdoors',
        2: 'Driving a Car',
        3: 'Lunch Break',
        4: 'Playing Table Soccer',
        5: 'Sitting and Reading',
        6: 'Transition',
        7: 'Walking',
        8: 'Working at Desk'
    }

    print("\n--- Loading Dataset ---")
    ANNOTATION_COLUMN = 'ActivityDescr'
    STATIC_FEATURES = ['HeartRate', 'Age', 'Gender', 'Height', 'Weight']

    full_dataset = ActivityDataset(
        data_path=DATA_PATH,
        annotation_col=ANNOTATION_COLUMN,
        statics=STATIC_FEATURES,
        transform=accel_to_mel,
        sample_rate=SR,
        sample_length_s=SEGMENT_DURATION_SEC,
        sliding_window_s=SLIDING_WINDOW_SEC
    )

    num_samples = len(full_dataset)
    if num_samples == 0:
        raise ValueError("Dataset is empty after initialization. Check data and annotations.")

    num_val = int(VAL_SPLIT * num_samples)
    num_train = num_samples - num_val
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    
    from transformers import Siglip2Processor, SiglipModel
    model_id = "google/siglip2-base-patch16-224"
    processor = Siglip2Processor.from_pretrained(model_id)
    model = SiglipModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16  # Half-precision for GPU
    ).to("cuda")

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # ----- Initialize MusicGen Text Embedding System -----
    from transformers import AutoTokenizer, AutoModel

    # Initialize T5 tokenizer and model
    muse_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    muse_text_encoder = AutoModel.from_pretrained("t5-base").encoder
    muse_text_encoder.eval().to(DEVICE)

    # Freeze T5 model parameters
    for param in muse_text_encoder.parameters():
        param.requires_grad = False

    # --- Training ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = train_encoder(
        model=model,
        processor=processor,
        activity_labels=activity_labels,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        muse_tokenizer=muse_tokenizer,
        muse_text_encoder=muse_text_encoder,
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
        train_path = project_root / 'dev' / 'encoder_scratch' / 'siglip_training'
        plot_training_history(history, train_path)
    except ImportError:
        print("\nMatplotlib not found. Skipping history plot.")
    except Exception as e:
        print(f"\nCould not generate plot. Error: {e}")
    
    print("\nScript finished.")