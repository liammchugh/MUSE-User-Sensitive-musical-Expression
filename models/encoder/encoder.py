import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Lightweight CNN Model ---
class SimpleCNN(nn.Module):
    """
    A simple lightweight CNN for classification, optionally incorporating static features.
    """
    def __init__(self, num_classes, num_statics, img_size=64, static_hidden_dim=32):
        super().__init__()
        self.num_statics = num_statics
        # Input: (B, 3, img_size, img_size)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 16, img_size/2, img_size/2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 32, img_size/4, img_size/4)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 64, img_size/8, img_size/8)
        )

        self.flatten = nn.Flatten()

        # Calculate flattened size dynamically after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            dummy_output = self.conv_block3(self.conv_block2(self.conv_block1(dummy_input)))
            self.flattened_conv_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]

        # Feed-forward network for static features
        if self.num_statics > 0:
            self.static_ffn = nn.Sequential(
                nn.Linear(num_statics, static_hidden_dim),
                nn.ReLU(),
                # nn.Dropout(0.2), # Optional dropout for static features
                nn.Linear(static_hidden_dim, static_hidden_dim), # Second layer
                nn.ReLU()
            )
            combined_features_size = self.flattened_conv_size + static_hidden_dim
        else:
            self.static_ffn = None
            combined_features_size = self.flattened_conv_size


        # Classifier part adjusted for combined features
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_static=None):
        """
        Forward pass.

        Args:
            x_img (torch.Tensor): Input image tensor (B, 3, H, W).
            x_static (torch.Tensor, optional): Input static features tensor (B, num_statics).
                                                Required if num_statics > 0 during init. Defaults to None.

        Returns:
            torch.Tensor: Output logits (B, num_classes).
        """
        # Process image features
        x_img = self.conv_block1(x_img)
        x_img = self.conv_block2(x_img)
        x_img = self.conv_block3(x_img)
        x_img_flat = self.flatten(x_img) # (B, flattened_conv_size)

        # Process static features if available
        if self.num_statics > 0:
            if x_static is None:
                raise ValueError("Static features (x_static) must be provided when num_statics > 0.")
            if x_static.shape[1] != self.num_statics:
                 raise ValueError(f"Expected {self.num_statics} static features, but got {x_static.shape[1]}.")
            x_static_processed = self.static_ffn(x_static) # (B, static_hidden_dim)

            # Concatenate features
            x_combined = torch.cat((x_img_flat, x_static_processed), dim=1) # (B, combined_features_size)
        else:
            # Only use image features if no static features are configured
            x_combined = x_img_flat

        # Classify combined features
        logits = self.classifier(x_combined)
        return logits
