import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms # Used for standard transforms if Albumentations is not preferred
import cv2
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Tuple, Dict, List, Optional
import warnings

# Suppress specific warnings, e.g., from Albumentations or PyTorch
warnings.filterwarnings('ignore')

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import Albumentations, provide fallback if not available
try:
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    logger.info("Albumentations found and will be used for data augmentation.")
except ImportError:
    logger.warning("Albumentations not found. Using torchvision transforms as fallback. "
                   "Install with: pip install albumentations opencv-python-headless")
    A = None # Set A to None to indicate Albumentations is not available

# ---------------------------
# Global Class Mapping and Colormap for Segmentation
# ---------------------------

CLASS_MAPPING = {
    "background": 0,
    "septum": 1, "s": 1,
    "right atrium": 2, "ad": 2,
    "left atrium": 3, "as": 3,
    "aorta": 4, "ao": 4,
    "right ventricle": 5, "vd": 5,
    "left ventricle": 6, "vs": 6,
    "aorta left ventricle outflow track open valves": 7, "aolvotov": 7,
    "aorta left ventricle outflow tract closed valves": 8, "aolvotcv": 8,
    "pulmonary artery": 9, "pa": 9,
    "3-vessels view": 10, "3vv": 10,
    "superior vena cava": 11, "svc": 11
}
ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

# Colormap for visualization (ensure it has enough colors for all classes)
# RGB tuples (0-1 range for matplotlib)
CLASS_COLORMAP = {
    0: (0.0, 0.0, 0.0),      # Background (Black)
    1: (0.8, 0.2, 0.2),      # Septum (Reddish)
    2: (0.2, 0.8, 0.2),      # Right Atrium (Greenish)
    3: (0.2, 0.2, 0.8),      # Left Atrium (Blueish)
    4: (0.8, 0.8, 0.2),      # Aorta (Yellowish)
    5: (0.2, 0.8, 0.8),      # Right Ventricle (Cyanish)
    6: (0.8, 0.2, 0.8),      # Left Ventricle (Magentaish)
    7: (0.5, 0.5, 0.0),      # AoLVOTOV (Olive)
    8: (0.0, 0.5, 0.5),      # AoLVOTCV (Teal)
    9: (0.5, 0.0, 0.5),      # Pulmonary Artery (Purple)
    10: (0.7, 0.4, 0.0),     # 3-Vessels View (Orange)
    11: (0.4, 0.7, 0.0)      # SVC (Lime Green)
}


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EfficientAttentionUNet(nn.Module):
    """
    Enhanced U-Net with attention mechanisms and EfficientNet-inspired blocks.
    Optimized for fetal echocardiography segmentation with 11 anatomical structures.
    It includes deep supervision for improved training stability and performance.
    """
    
    def __init__(self, in_channels=3, num_classes=12, base_filters=64):
        """
        Initializes the EfficientAttentionUNet model.

        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            num_classes (int): Number of segmentation classes (including background).
            base_filters (int): Base number of filters for the first encoder block.
        """
        super(EfficientAttentionUNet, self).__init__()
        
        # Encoder (Downsampling path)
        # Each encoder block consists of two Conv2d layers followed by BatchNorm, ReLU, and MaxPool.
        self.encoder1 = self._make_encoder_block(in_channels, base_filters)
        self.encoder2 = self._make_encoder_block(base_filters, base_filters * 2)
        self.encoder3 = self._make_encoder_block(base_filters * 2, base_filters * 4)
        self.encoder4 = self._make_encoder_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck with ASPP (Atrous Spatial Pyramid Pooling)
        # ASPP captures multi-scale context, crucial for varying object sizes.
        self.bottleneck = ASPPModule(base_filters * 8, base_filters * 16)
        
        # Decoder (Upsampling path) with skip connections and attention gates
        # Each decoder block uses ConvTranspose2d for upsampling, followed by Conv2d, BatchNorm, ReLU.
        # Attention gates are applied to skip connections to selectively pass relevant features.
        # Note: Input channels for decoders are sum of upsampled feature map and skip connection (after attention)
        self.decoder4 = self._make_decoder_block(base_filters * 16, base_filters * 8)
        self.attention4 = AttentionGate(base_filters * 8, base_filters * 8)
        
        self.decoder3 = self._make_decoder_block(base_filters * 16, base_filters * 4) 
        self.attention3 = AttentionGate(base_filters * 4, base_filters * 4)
        
        self.decoder2 = self._make_decoder_block(base_filters * 8, base_filters * 2)
        self.attention2 = AttentionGate(base_filters * 2, base_filters * 2)
        
        self.decoder1 = self._make_decoder_block(base_filters * 4, base_filters)
        self.attention1 = AttentionGate(base_filters, base_filters)
        
        # Final classification layer: maps concatenated features to num_classes
        self.final_conv = nn.Conv2d(base_filters * 2, num_classes, kernel_size=1)
        
        # Deep supervision (auxiliary outputs at different decoder stages)
        # These help stabilize training by providing additional loss signals.
        self.aux_conv4 = nn.Conv2d(base_filters * 8, num_classes, kernel_size=1)
        self.aux_conv3 = nn.Conv2d(base_filters * 4, num_classes, kernel_size=1)
        self.aux_conv2 = nn.Conv2d(base_filters * 2, num_classes, kernel_size=1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Helper function to create an encoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels), # Add SEBlock to encoder
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Helper function to create a decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels) # Add SEBlock to decoder
        )
    
    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            tuple or torch.Tensor: If in training mode, returns main output and auxiliary outputs.
                                   Otherwise, returns only the main output.
        """
        # Encoder path
        # The MaxPool is the last operation of _make_encoder_block, so enc1, enc2, enc3, enc4 are already downsampled.
        enc1 = self.encoder1(x)  # Output: 64 channels, 1/2 resolution
        enc2 = self.encoder2(enc1)  # Output: 128 channels, 1/4 resolution
        enc3 = self.encoder3(enc2)  # Output: 256 channels, 1/8 resolution
        enc4 = self.encoder4(enc3)  # Output: 512 channels, 1/16 resolution
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # Output: 1024 channels (if base_filters=64), 1/16 resolution
        
        # Decoder path with attention and skip connections
        # Upconv layers are added before the decoder blocks to upsample features
        dec4_up = nn.ConvTranspose2d(bottleneck.shape[1], enc4.shape[1], kernel_size=2, stride=2)(bottleneck)
        att4 = self.attention4(dec4_up, enc4) # Apply attention to the skip connection
        dec4_concat = torch.cat([dec4_up, att4], dim=1) # Concatenate upsampled decoder output with attended skip
        dec4 = self.decoder4(dec4_concat)
        aux4 = self.aux_conv4(att4) # Auxiliary output from this stage
        
        dec3_up = nn.ConvTranspose2d(dec4.shape[1], enc3.shape[1], kernel_size=2, stride=2)(dec4)
        att3 = self.attention3(dec3_up, enc3)
        dec3_concat = torch.cat([dec3_up, att3], dim=1)
        dec3 = self.decoder3(dec3_concat)
        aux3 = self.aux_conv3(att3)
        
        dec2_up = nn.ConvTranspose2d(dec3.shape[1], enc2.shape[1], kernel_size=2, stride=2)(dec3)
        att2 = self.attention2(dec2_up, enc2)
        dec2_concat = torch.cat([dec2_up, att2], dim=1)
        dec2 = self.decoder2(dec2_concat)
        aux2 = self.aux_conv2(att2)
        
        dec1_up = nn.ConvTranspose2d(dec2.shape[1], enc1.shape[1], kernel_size=2, stride=2)(dec2)
        att1 = self.attention1(dec1_up, enc1)
        dec1_concat = torch.cat([dec1_up, att1], dim=1)
        dec1 = self.decoder1(dec1_concat)
        
        # Final output
        output = self.final_conv(dec1)
        
        if self.training:
            # Return main output and auxiliary outputs for deep supervision
            return output, aux4, aux3, aux2
        else:
            return output

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.
    Captures multi-scale contextual information by applying atrous convolutions
    with different rates, and global average pooling.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initializes the ASPP module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for the module.
        """
        super(ASPPModule, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        # 3x3 atrous convolutions with different dilation rates
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=18, dilation=18)
        
        # Global average pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1) # Output size 1x1
        self.global_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        
        # Final 1x1 convolution to combine all branches
        # The total input channels will be (out_channels // 4) * 5
        self.final_conv = nn.Conv2d((out_channels // 4) * 5, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Forward pass through the ASPP module.

        Args:
            x (torch.Tensor): Input tensor from the encoder.

        Returns:
            torch.Tensor: Output tensor with multi-scale features.
        """
        size = x.shape[2:] # Get original height and width for interpolation
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        # Global average pooling branch
        x5 = self.global_pool(x)
        x5 = self.global_conv(x5)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True) # Upsample to original size
        
        # Concatenate all branches and apply final convolution
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.final_conv(x)
        x = self.dropout(x)
        
        return x

class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    It learns to focus on relevant regions in the skip connection features
    based on the features from the coarser (gate) pathway.
    """
    
    def __init__(self, gate_channels, skip_channels):
        """
        Initializes the Attention Gate.

        Args:
            gate_channels (int): Number of channels in the gating signal (from decoder).
            skip_channels (int): Number of channels in the skip connection (from encoder).
        """
        super(AttentionGate, self).__init__()
        
        # 1x1 convolution on the gating signal
        self.gate_conv = nn.Conv2d(gate_channels, skip_channels, kernel_size=1)
        # 1x1 convolution on the skip connection features
        self.skip_conv = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
        # Final 1x1 convolution to produce attention coefficients
        self.attention_conv = nn.Conv2d(skip_channels, 1, kernel_size=1)
        
    def forward(self, gate, skip):
        """
        Forward pass through the Attention Gate.

        Args:
            gate (torch.Tensor): Gating signal (from a deeper decoder stage).
            skip (torch.Tensor): Skip connection features (from an encoder stage).

        Returns:
            torch.Tensor: Attended skip connection features.
        """
        # Resize gate signal to match skip connection spatial dimensions
        gate_resized = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply 1x1 convolutions
        gate_conv = self.gate_conv(gate_resized)
        skip_conv = self.skip_conv(skip)
        
        # Combine and apply ReLU, then 1x1 conv, and sigmoid to get attention coefficients
        attention = torch.sigmoid(self.attention_conv(F.relu(gate_conv + skip_conv)))
        
        # Apply attention coefficients to the skip connection
        attended_skip = skip * attention
        
        return attended_skip

class FetalHeartDataset(Dataset):
    """
    Dataset class for fetal heart ultrasound images with segmentation masks.
    Supports loading images and corresponding masks for segmentation tasks.
    """
    
    def __init__(self, images_dir, masks_dir=None, transform=None):
        """
        Initializes the FetalHeartDataset for segmentation.

        Args:
            images_dir (str or Path): Directory containing the ultrasound images.
            masks_dir (str or Path, optional): Directory containing the corresponding segmentation masks.
                                               Required for 'segmentation' mode.
            transform (albumentations.Compose or torchvision.transforms.Compose, optional):
                      Data augmentation and preprocessing transforms.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        
        # Collect image paths
        self.image_paths = sorted(list(self.images_dir.rglob('*.png')) +
                                  list(self.images_dir.rglob('*.jpg')) +
                                  list(self.images_dir.rglob('*.jpeg')))
        
        if not self.masks_dir:
            logger.error("masks_dir must be provided for segmentation dataset.")
            self.image_paths = [] # Clear paths if masks_dir is missing
            return

        # Filter image paths to ensure corresponding masks exist
        # Assumes mask filename is image_filename_mask.png or similar
        valid_image_paths = []
        self.mask_paths = []
        for img_path in self.image_paths:
            # Example: image.png -> image_mask.png (adjust naming convention if needed)
            mask_filename = img_path.stem + "_mask.png" # Common convention
            # If masks are in a parallel structure mirroring image_dir, adjust this:
            # mask_path = self.masks_dir / img_path.relative_to(self.images_dir).parent / mask_filename
            
            # Assuming masks_dir directly contains masks named like original image stems + "_mask.png"
            mask_path = self.masks_dir / mask_filename
            
            if mask_path.exists():
                valid_image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                logger.warning(f"No corresponding mask found for {img_path}. Skipping.")
        
        self.image_paths = valid_image_paths
        
        logger.info(f"Loaded {len(self.image_paths)} images for segmentation.")
        if not self.image_paths:
            logger.error("No valid image-mask pairs found. Check your data paths and naming conventions.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retrieves an image-mask pair and applies transformations.

        Returns:
            tuple: (image_tensor, mask_tensor, image_stem)
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image (RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (Grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        
        if self.transform:
            if A is not None: # Use Albumentations if available
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else: # Fallback to torchvision (requires manual numpy to tensor conversion)
                # For torchvision, image needs to be PIL Image or torch.Tensor already
                # and mask needs separate handling. This is a simplified fallback.
                image = Image.fromarray(image)
                image = self.transform(image)
                # Mask transformation needs to be handled carefully if not using A.
                # For simplicity, if A is not used, mask is just converted to tensor.
                mask = torch.from_numpy(mask).long()
        
        if A is None: # Manual conversion to tensor if Albumentations ToTensorV2 is not used
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask, str(img_path.stem)

class CombinedLoss(nn.Module):
    """
    Combined loss function for segmentation: Dice Loss + Focal Loss + Deep Supervision.
    Optimized for class imbalance in medical segmentation.
    """
    
    def __init__(self, num_classes=12, alpha=0.25, gamma=2.0, dice_weight=0.7, 
                 focal_weight=0.3, deep_supervision_weights=[0.5, 0.3, 0.2]):
        """
        Initializes the CombinedLoss.

        Args:
            num_classes (int): Number of segmentation classes.
            alpha (float): Alpha parameter for Focal Loss (weighting of positive/negative examples).
            gamma (float): Gamma parameter for Focal Loss (focusing parameter).
            dice_weight (float): Weight for Dice Loss in the combined loss.
            focal_weight (float): Weight for Focal Loss in the combined loss.
            deep_supervision_weights (list): Weights for auxiliary losses in deep supervision.
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.deep_supervision_weights = deep_supervision_weights
        
    def dice_loss(self, pred, target):
        """
        Calculates multi-class Dice Loss.

        Args:
            pred (torch.Tensor): Predicted logits from the model (N, C, H, W).
            target (torch.Tensor): Ground truth masks (N, H, W).

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        smooth = 1e-5
        pred_softmax = F.softmax(pred, dim=1) # Convert logits to probabilities
        
        dice_scores = []
        for i in range(self.num_classes):
            pred_i = pred_softmax[:, i, :, :] # Probability map for class i
            target_i = (target == i).float() # Binary mask for class i
            
            intersection = (pred_i * target_i).sum(dim=(1, 2))
            union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))
            
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)
        return 1 - dice_scores.mean() # Return 1 - mean Dice score (to minimize)
    
    def focal_loss(self, pred, target):
        """
        Calculates multi-class Focal Loss.

        Args:
            pred (torch.Tensor): Predicted logits from the model (N, C, H, W).
            target (torch.Tensor): Ground truth masks (N, H, W).

        Returns:
            torch.Tensor: Scalar Focal loss.
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        # Probability of the predicted class
        pt = torch.exp(-ce_loss)
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs, target):
        """
        Computes the combined loss.

        Args:
            outputs (tuple or torch.Tensor): Model outputs. If tuple, assumes deep supervision
                                             (main_output, aux4, aux3, aux2).
                                             Otherwise, a single output tensor.
            target (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Total combined loss.
        """
        if isinstance(outputs, tuple):
            # Deep supervision mode: calculate main loss and auxiliary losses
            main_output, aux4, aux3, aux2 = outputs
            
            # Resize auxiliary outputs to match target size for loss calculation
            target_size = target.shape[1:] # (H, W) for target
            aux4_resized = F.interpolate(aux4, size=target_size, mode='bilinear', align_corners=True)
            aux3_resized = F.interpolate(aux3, size=target_size, mode='bilinear', align_corners=True)
            aux2_resized = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=True)
            
            # Main loss calculation
            main_dice = self.dice_loss(main_output, target)
            main_focal = self.focal_loss(main_output, target)
            main_loss = self.dice_weight * main_dice + self.focal_weight * main_focal
            
            # Auxiliary losses calculation
            aux_losses = []
            for aux_output, weight in zip([aux4_resized, aux3_resized, aux2_resized], 
                                        self.deep_supervision_weights):
                aux_dice = self.dice_loss(aux_output, target)
                aux_focal = self.focal_loss(aux_output, target)
                aux_loss = self.dice_weight * aux_dice + self.focal_weight * aux_focal
                aux_losses.append(weight * aux_loss)
            
            total_loss = main_loss + sum(aux_losses)
            return total_loss
        
        else:
            # Single output mode (e.g., during inference or if deep supervision is off)
            dice = self.dice_loss(outputs, target)
            focal = self.focal_loss(outputs, target)
            return self.dice_weight * dice + self.focal_weight * focal

class SegmentationTrainer:
    """
    Comprehensive trainer for fetal heart segmentation model.
    Includes training loop, validation, checkpointing, and metrics tracking.
    """
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-4, weight_decay=1e-5):
        """
        Initializes the SegmentationTrainer.

        Args:
            model (nn.Module): The segmentation model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to train on ('cuda' or 'cpu').
            learning_rate (float): Initial learning rate for the optimizer.
            weight_decay (float): L2 regularization parameter for the optimizer.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer: AdamW is a good choice for deep learning models
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                   weight_decay=weight_decay)
        # Learning Rate Scheduler: CosineAnnealingWarmRestarts for cyclical LR
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # Loss function
        self.criterion = CombinedLoss() # Using the custom combined loss
        
        # Metrics tracking lists
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
        # Best model tracking for checkpointing
        self.best_val_dice = 0.0
        self.best_model_state = None
    
    def calculate_dice_score(self, pred, target, num_classes=12):
        """
        Calculates multi-class Dice score.

        Args:
            pred (torch.Tensor): Predicted logits from the model (N, C, H, W).
            target (torch.Tensor): Ground truth masks (N, H, W).
            num_classes (int): Total number of classes.

        Returns:
            float: Mean Dice score across all classes.
        """
        pred_softmax = F.softmax(pred, dim=1)
        pred_classes = torch.argmax(pred_softmax, dim=1) # Get predicted class for each pixel
        
        dice_scores = []
        for i in range(num_classes):
            pred_i = (pred_classes == i).float() # Binary mask for predicted class i
            target_i = (target == i).float() # Binary mask for ground truth class i
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            if union > 0:
                dice = (2 * intersection) / union
            else:
                dice = 1.0  # If both pred and target are empty for a class, it's a perfect score
            
            dice_scores.append(dice.item())
        
        return np.mean(dice_scores)
    
    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0.0
        
        # Use tqdm for a progress bar during training
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients, perform forward pass, calculate loss, backward pass, and optimize
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(self.train_loader) # Average loss for the epoch
    
    def validate(self):
        """Validates the model on the validation set."""
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        total_dice = 0.0
        
        with torch.no_grad(): # Disable gradient calculation for validation
            for images, masks, _ in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks) # Calculate loss
                
                # If deep supervision is active, use the main output for metrics
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                dice_score = self.calculate_dice_score(main_output, masks) # Calculate Dice score
                
                total_loss += loss.item()
                total_dice += dice_score
        
        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        return avg_loss, avg_dice
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """
        Runs the complete training loop for the specified number of epochs.

        Args:
            num_epochs (int): Total number of epochs to train.
            save_dir (str): Directory to save model checkpoints.
        """
        os.makedirs(save_dir, exist_ok=True) # Create save directory if it doesn't exist
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate the model
            val_loss, val_dice = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_dice_scores.append(val_dice)
            
            logger.info(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save the best model based on validation Dice score
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_model_state = self.model.state_dict().copy() # Save a copy of the best state
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, os.path.join(save_dir, 'best_model.pth'))
                logger.info(f"Saved new best model with Dice: {val_dice:.4f}")
            
            # Save a checkpoint periodically (e.g., every 10 epochs)
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_dice_scores': self.val_dice_scores
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                logger.info(f"Saved checkpoint for epoch {epoch+1}")
        
        logger.info(f"Training completed. Final best validation Dice: {self.best_val_dice:.4f}")
        
        # Load the best model state back into the model at the end of training
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state for final evaluation/inference.")
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plots the training and validation loss, and validation Dice score over epochs.

        Args:
            save_path (str): Path to save the generated plot.
        """
        if not self.train_losses or not self.val_losses or not self.val_dice_scores:
            logger.warning("No training history to plot. Run training first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Dice score plot
        ax2.plot(epochs, self.val_dice_scores, 'g-', label='Validation Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Validation Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to: {save_path}")
        plt.show()

class FetalCHDClassifier(nn.Module):
    """
    Classification model for CHD detection (Normal/ASD/VSD).
    It uses the encoder part of a pre-trained segmentation model as a feature extractor
    and adds a classification head on top.
    """
    
    def __init__(self, segmentation_model, num_classes=3, freeze_backbone=True):
        """
        Initializes the FetalCHDClassifier.

        Args:
            segmentation_model (EfficientAttentionUNet): An instance of the segmentation model
                                                         to use as a backbone.
            num_classes (int): Number of classification output classes (e.g., 3 for Normal, ASD, VSD).
            freeze_backbone (bool): If True, freezes the weights of the segmentation backbone.
        """
        super(FetalCHDClassifier, self).__init__()
        
        # Use segmentation model encoder as feature extractor
        self.backbone = segmentation_model
        self.freeze_backbone = freeze_backbone # Store for conditional grad enabling
        
        if freeze_backbone:
            # Freeze backbone parameters to prevent them from being updated during classification training
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Segmentation backbone parameters frozen for classification training.")
        else:
            logger.info("Segmentation backbone parameters are NOT frozen for classification training.")
        
        # Global average pooling to reduce spatial dimensions to a single feature vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head: A series of fully connected layers with dropout and ReLU
        # The input size to the first linear layer should match the output channels of the bottleneck.
        # Assuming bottleneck has base_filters * 16 channels (e.g., 64 * 16 = 1024)
        bottleneck_output_channels = segmentation_model.bottleneck.final_conv.out_channels
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(bottleneck_output_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes) # Final layer maps to the number of classification classes
        )
    
    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Logits for classification.
        """
        # Extract features using segmentation backbone's encoder and bottleneck
        # Use torch.set_grad_enabled to control gradient calculation for the backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            enc1 = self.backbone.encoder1(x)
            enc2 = self.backbone.encoder2(enc1)
            enc3 = self.backbone.encoder3(enc2)
            enc4 = self.backbone.encoder4(enc3)
            features = self.backbone.bottleneck(enc4)
        
        # Global pooling and classification
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1) # Flatten the pooled features
        logits = self.classifier(pooled) # Pass through the classification head
        
        return logits

def create_data_transforms(seg_target_size=(256, 256), cls_target_size=(224, 224)):
    """
    Creates data augmentation transforms for both segmentation and classification tasks.
    Uses Albumentations if available, otherwise a basic torchvision transform.

    Args:
        seg_target_size (tuple): Target size (H, W) for segmentation images.
        cls_target_size (tuple): Target size (H, W) for classification images.

    Returns:
        dict: A dictionary containing different transform pipelines.
    """
    if A is not None:
        # For segmentation training (image and mask augmentations)
        train_transform_seg = A.Compose([
            A.Resize(seg_target_size[0], seg_target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 3), p=0.3), # blur_limit must be a tuple or single odd int
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2() # Converts image to PyTorch tensor and scales to [0,1]
        ])
        
        val_transform_seg = A.Compose([
            A.Resize(seg_target_size[0], seg_target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # For classification training (image augmentations only)
        train_transform_cls = A.Compose([
            A.Resize(cls_target_size[0], cls_target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        val_transform_cls = A.Compose([
            A.Resize(cls_target_size[0], cls_target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        logger.warning("Albumentations not available. Using basic torchvision transforms. "
                       "Augmentations will be limited.")
        # Fallback to torchvision transforms if Albumentations is not installed
        train_transform_seg = transforms.Compose([
            transforms.ToPILImage(), # Convert numpy array to PIL Image
            transforms.Resize(seg_target_size),
            transforms.ToTensor(), # Converts to [0,1] and (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform_seg = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(seg_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_transform_cls = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cls_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform_cls = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(cls_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return {
        'train_seg': train_transform_seg,
        'val_seg': val_transform_seg,
        'train_cls': train_transform_cls,
        'val_cls': val_transform_cls
    }

class InferenceEngine:
    """
    Inference engine for generating segmentation predictions on new data.
    Can be used for creating rough labels on first trimester images using a trained
    second trimester segmentation model. Also includes visualization and metric calculation.
    """
    
    def __init__(self, model_path, num_classes=NUM_CLASSES, target_size=(384, 384), device='cuda'):
        """
        Initializes the InferenceEngine.

        Args:
            model_path (str): Path to the trained segmentation model checkpoint (.pth file).
            num_classes (int): Number of classes the model was trained to predict.
            target_size (tuple): Input size (H, W) the model expects.
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = EfficientAttentionUNet(num_classes=num_classes)
        
        # Load trained model state dictionary
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle cases where model_state_dict might be nested (e.g., if saved with 'model_state_dict' key)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint) # Assume checkpoint is just the state_dict
            self.model.to(device)
            self.model.eval() # Set model to evaluation mode
            logger.info(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model checkpoint not found at {model_path}. Please check the path.")
            self.model = None
            return
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            self.model = None
            return
        
        self.image_size = target_size # Store the model's input size
        # Define preprocessing transform for inference
        if A is not None:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_AREA),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        logger.info(f"Inference preprocessing pipeline: Resize to {self.image_size} and Normalize.")

    def _load_image(self, image_path: Path):
        """Loads an image using OpenCV, with PIL fallback for robustness."""
        image = cv2.imread(str(image_path))
        if image is None:
            try:
                # Fallback to PIL for more robust image loading
                img_pil = Image.open(image_path)
                image = np.array(img_pil)
                if image.ndim == 2: # Grayscale to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.ndim == 3 and image.shape[2] == 4: # RGBA to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.ndim == 3 and image.shape[2] == 3: # RGB to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    raise RuntimeError(f"Unsupported image format or dimensions for {image_path}")
                logger.warning(f"Using PIL fallback to load image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to load image {image_path} with both cv2 and PIL: {e}")
                return None
        return image

    def predict_segmentation(self, image_path: Path):
        """
        Performs inference on a single image and returns original image (RGB),
        predicted mask, pixel-wise confidence, and raw probabilities.
        """
        original_image_bgr = self._load_image(image_path)
        if original_image_bgr is None:
            return None, None, None, None

        # Convert to RGB for Albumentations and display
        original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        if A is not None:
            transformed = self.transform(image=original_image_rgb)
            input_image_tensor = transformed['image'].unsqueeze(0).float()
        else:
            image_pil = Image.fromarray(original_image_rgb)
            input_image_tensor = self.transform(image_pil).unsqueeze(0).float()
        
        input_image_tensor = input_image_tensor.to(self.device)

        with torch.no_grad():
            output_logits = self.model(input_image_tensor)
            # If deep supervision was used during training, output might be a tuple.
            # Take the main output for inference.
            if isinstance(output_logits, tuple):
                output_logits = output_logits[0]

            # Resize output logits to the original image shape for overlay/metrics
            # This is crucial if original_image_rgb size is different from self.image_size (input to model)
            output_logits_resized = F.interpolate(output_logits, 
                                                  size=original_image_rgb.shape[:2], 
                                                  mode='bilinear', 
                                                  align_corners=False)

            # Apply softmax to get probabilities
            probabilities = torch.softmax(output_logits_resized, dim=1).squeeze().cpu().numpy() # C x H x W

            # Get predicted mask (class ID for each pixel)
            predicted_mask = np.argmax(probabilities, axis=0) # H x W

            # Get confidence for the predicted class at each pixel
            # This is the probability of the *predicted* class for that pixel
            confidence_map = np.max(probabilities, axis=0) # H x W
        
        return original_image_rgb, predicted_mask, confidence_map, probabilities

    def visualize_prediction(self, original_image: np.ndarray, predicted_mask: np.ndarray, confidence_map: np.ndarray, image_name: str, save_dir: Path):
        """
        Plots the original image, predicted mask, and an overlay.
        Predicted mask is colored using CLASS_COLORMAP.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original Image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image: {image_name}')
        axes[0].axis('off')

        # 2. Predicted Mask
        # Create a blank RGB image to apply colors based on class IDs
        colored_mask = np.zeros((*predicted_mask.shape, 3), dtype=np.float32)
        for class_id, color in CLASS_COLORMAP.items():
            colored_mask[predicted_mask == class_id] = color

        axes[1].imshow(colored_mask)
        axes[1].set_title('Predicted Segmentation Mask')
        axes[1].axis('off')

        # Create a legend
        handles = [plt.Line2D([0], [0], color=color, lw=4, label=ID_TO_CLASS[id]) 
                   for id, color in CLASS_COLORMAP.items()]
        fig.legend(handles=handles, title="Classes", loc='center right', bbox_to_anchor=(1.1, 0.5))

        # 3. Confidence Map (e.g., for predicted class)
        # Using a grayscale colormap for confidence (0 to 1)
        im_conf = axes[2].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Pixel-wise Prediction Confidence')
        axes[2].axis('off')
        fig.colorbar(im_conf, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
        
        save_path = save_dir / f"{image_name}_prediction.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Prediction visualization saved to: {save_path}")
        plt.show()
        plt.close(fig)

    def calculate_metrics(self, ground_truth_mask: np.ndarray, predicted_mask: np.ndarray):
        """
        Calculates and prints metrics if ground truth is available.
        Uses jaccard_score (IoU).
        """
        if ground_truth_mask is None:
            logger.info("Ground truth mask not provided. Skipping metric calculation.")
            return

        gts_flat = ground_truth_mask.flatten()
        preds_flat = predicted_mask.flatten()

        # Ensure labels are explicitly passed to jaccard_score covering all classes (0 to NUM_CLASSES-1)
        # and unique values in GT and Preds are within this range
        
        # Filter out any unexpected high values in GT or Preds before computing metrics
        valid_gts_flat = gts_flat[gts_flat < NUM_CLASSES]
        valid_preds_flat = preds_flat[preds_flat < NUM_CLASSES]

        # Use only the common indices where both are valid for the jaccard_score
        # Create a boolean mask for valid labels within the expected range
        valid_indices_gt = (gts_flat >= 0) & (gts_flat < NUM_CLASSES)
        valid_indices_pred = (preds_flat >= 0) & (preds_flat < NUM_CLASSES)
        combined_valid_indices = valid_indices_gt & valid_indices_pred
        
        if not np.any(combined_valid_indices):
            logger.warning("No valid pixels for metric calculation (GT or Preds out of class range).")
            return
            
        gts_for_iou = gts_flat[combined_valid_indices]
        preds_for_iou = preds_flat[combined_valid_indices]

        # Use only labels that are actually present in the valid data
        active_labels = np.unique(np.concatenate((gts_for_iou, preds_for_iou)))
        active_labels = active_labels[active_labels < NUM_CLASSES] # Ensure they are within NUM_CLASSES

        if len(active_labels) == 0:
            logger.warning("No active labels (excluding out-of-range) for IoU calculation.")
            return

        # Calculate macro IoU over all possible classes (0 to NUM_CLASSES-1)
        iou_macro_all = jaccard_score(gts_for_iou, preds_for_iou, average='macro', zero_division=0, labels=np.arange(NUM_CLASSES))
        
        # Calculate per-class IoU for all possible classes
        iou_per_class_raw = jaccard_score(gts_for_iou, preds_for_iou, average=None, zero_division=0, labels=np.arange(NUM_CLASSES))

        # Foreground IoU (excluding background, class ID 0)
        foreground_labels = np.array([lbl for lbl in np.arange(NUM_CLASSES) if lbl != 0])
        iou_foreground = 0.0
        if len(foreground_labels) > 0:
            # Calculate foreground IoU by averaging IoU of non-background classes
            iou_foreground_per_class = jaccard_score(gts_for_iou, preds_for_iou, average=None, zero_division=0, labels=foreground_labels)
            if len(iou_foreground_per_class) > 0:
                iou_foreground = np.mean(iou_foreground_per_class)
        
        logger.info(f"\n--- Inference Metrics ---")
        logger.info(f"  Macro mIoU (All Classes): {iou_macro_all:.4f}")
        logger.info(f"  Foreground mIoU (Mean of non-background classes): {iou_foreground:.4f}")
        logger.info("  Per-Class IoU:")
        for class_id in range(NUM_CLASSES):
            class_name = ID_TO_CLASS.get(class_id, f"Class {class_id}")
            logger.info(f"    '{class_name}': {iou_per_class_raw[class_id]:.4f}")


def _load_labelme_annotations_for_gt(json_path: Path, shape: Tuple[int, int]):
    """
    Helper function to load a LabelMe JSON annotation and convert it into a
    segmentation mask (numpy array) with class IDs as pixel values.

    Args:
        json_path (Path): Path to the LabelMe JSON file.
        shape (Tuple[int, int]): Desired (height, width) of the output mask.

    Returns:
        numpy.ndarray: The generated ground truth mask, or None if loading fails.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shp in data.get('shapes', []):
            label = str(shp['label']).strip().lower()
            if label in CLASS_MAPPING:
                cls_id = CLASS_MAPPING[label]
                if cls_id >= NUM_CLASSES or cls_id < 0:
                    logger.warning(f"GT: Class ID {cls_id} for label '{label}' in {json_path} out of range. Skipping.")
                    continue
                if not shp.get('points') or len(shp['points']) < 3:
                    logger.warning(f"GT: Malformed polygon in {json_path} for label '{label}'. Skipping.")
                    continue
                pts = np.array(shp['points'], np.int32)
                # Ensure points are within image bounds
                pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
                cv2.fillPoly(mask, [pts], cls_id)
            else:
                logger.warning(f"GT: Unknown label '{shp['label']}' in {json_path}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading GT annotation {json_path}: {e}")
        return None
    return mask


def main_segmentation_pipeline():
    """
    Main function to run the fetal CHD segmentation training and optional inference pipeline.
    Configurable via command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Fetal CHD Segmentation Training and Inference Pipeline.")
    
    # General arguments
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'predict_labels', 'inference_single_image'],
                        help="Mode of operation: 'train' for model training, 'predict_labels' for generating labels using a trained model, 'inference_single_image' for visualizing and evaluating a single image. Default: train.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Root directory for training/inference images (e.g., 'data/second_trimester/images').")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help=f"Number of segmentation classes (including background). Default: {NUM_CLASSES}.")
    parser.add_argument("--checkpoint_save_dir", type=str, default="segmentation_checkpoints",
                        help="Directory to save model checkpoints and training history plot. Default: segmentation_checkpoints.")
    
    # Training-specific arguments
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Root directory for training masks (e.g., 'data/second_trimester/masks'). Required for 'train' mode.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs. Default: 100.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and validation. Default: 8.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for the optimizer. Default: 1e-4.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (L2 regularization) for the optimizer. Default: 1e-5.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading. Default: 4.")
    parser.add_argument("--seg_target_size", type=int, nargs=2, default=[256, 256],
                        help="Target (height width) size for segmentation images during training. Default: 256 256.")
    
    # Inference-specific arguments (for 'predict_labels' and 'inference_single_image' modes)
    parser.add_argument("--inference_model_path", type=str, default=os.path.join("segmentation_checkpoints", "best_model.pth"),
                        help="Path to the trained model checkpoint for inference. Required for 'predict_labels' and 'inference_single_image' modes. Default: segmentation_checkpoints/best_model.pth.")
    parser.add_argument("--inference_target_size", type=int, nargs=2, default=[384, 384],
                        help="Target (height width) size for images during inference. Default: 384 384.")
    
    # Arguments specific to 'predict_labels' mode
    parser.add_argument("--predicted_masks_output_dir", type=str, default="data/first_trimester/predicted_masks",
                        help="Output directory for auto-generated masks during 'predict_labels' mode. Default: data/first_trimester/predicted_masks.")

    # Arguments specific to 'inference_single_image' mode
    parser.add_argument("--single_image_path", type=str, default=None,
                        help="Path to a single image for visualization and metric calculation in 'inference_single_image' mode.")
    parser.add_argument("--single_image_gt_json_path", type=str, default=None,
                        help="Path to the LabelMe JSON ground truth annotation for the single image. Optional for 'inference_single_image' mode.")
    parser.add_argument("--single_image_output_dir", type=str, default="inference_visualizations",
                        help="Output directory for visualizations of single image inference. Default: inference_visualizations.")


    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'train':
        if not args.masks_dir:
            logger.error("--masks_dir is required for 'train' mode.")
            return

        # Create transforms
        transforms_dict = create_data_transforms(seg_target_size=tuple(args.seg_target_size))
        
        # Create full dataset
        full_dataset = FetalHeartDataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            transform=transforms_dict['train_seg'] # Use train transform for the full dataset initially
        )
        
        if not full_dataset.image_paths:
            logger.error("No valid data found for training. Exiting.")
            return

        # Split into train and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Assign validation transform to the validation subset's underlying dataset
        # This is a common pattern when using random_split with custom datasets
        # Note: This modifies the transform for the entire original dataset,
        # which is then reflected in both train_dataset and val_dataset.
        # For more robust handling, consider creating separate Dataset instances
        # for train and val with their respective transforms.
        full_dataset.transform = transforms_dict['val_seg'] # Apply val transform to the original dataset

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True)
        
        # Create model
        model = EfficientAttentionUNet(in_channels=3, num_classes=args.num_classes)
        
        # Create trainer
        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Train model
        logger.info(f"Starting segmentation model training for {args.num_epochs} epochs...")
        trainer.train(num_epochs=args.num_epochs, save_dir=args.checkpoint_save_dir)
        
        # Plot training history
        trainer.plot_training_history(os.path.join(args.checkpoint_save_dir, 'segmentation_training_history.png'))
        
        logger.info("Segmentation model training completed!")

    elif args.mode == 'predict_labels':
        # Initialize inference engine
        inference_engine = InferenceEngine(
            model_path=args.inference_model_path,
            num_classes=args.num_classes,
            target_size=tuple(args.inference_target_size), # Use inference specific target size
            device=device
        )
        
        if inference_engine.model is None:
            logger.error("Inference engine could not be initialized. Exiting 'predict_labels' mode.")
            return

        # Generate predictions for specified images (e.g., first trimester images)
        logger.info(f"Generating rough segmentation labels for images in {args.images_dir}...")
        predictions = inference_engine.batch_predict(
            image_dir=args.images_dir,
            output_dir=args.predicted_masks_output_dir
        )
        
        logger.info(f"Generated predictions for {len(predictions)} images.")

    elif args.mode == 'inference_single_image':
        if not args.single_image_path:
            logger.error("--single_image_path is required for 'inference_single_image' mode.")
            return

        # Initialize inference engine
        inference_engine = InferenceEngine(
            model_path=args.inference_model_path,
            num_classes=args.num_classes,
            target_size=tuple(args.inference_target_size),
            device=device
        )

        if inference_engine.model is None:
            logger.error("Inference engine could not be initialized. Exiting 'inference_single_image' mode.")
            return

        single_image_path = Path(args.single_image_path)
        output_visualization_dir = Path(args.single_image_output_dir)
        output_visualization_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Performing inference on single image: {single_image_path}")
        original_image, predicted_mask, confidence_map, _ = inference_engine.predict_segmentation(single_image_path)

        if original_image is not None:
            image_name = single_image_path.stem
            inference_engine.visualize_prediction(original_image, predicted_mask, confidence_map, image_name, output_visualization_dir)

            # Optionally, load ground truth mask and calculate metrics
            ground_truth_mask = None
            if args.single_image_gt_json_path:
                gt_json_path = Path(args.single_image_gt_json_path)
                if gt_json_path.exists():
                    logger.info(f"Loading ground truth from: {gt_json_path}")
                    original_shape_for_gt = original_image.shape[:2] # Get H, W from the loaded image
                    ground_truth_mask = _load_labelme_annotations_for_gt(gt_json_path, original_shape_for_gt)
                else:
                    logger.warning(f"Ground truth JSON not found at {gt_json_path}. Skipping metric calculation.")
            
            inference_engine.calculate_metrics(ground_truth_mask, predicted_mask)
        else:
            logger.error(f"Failed to process single image {single_image_path}.")
    
    logger.info("Segmentation pipeline completed successfully!")

if __name__ == "__main__":
    # Ensure Albumentations is installed if not using fallback
    if A is None:
        logger.error("Albumentations library is highly recommended for data augmentation. "
                     "Please install it using 'pip install albumentations opencv-python-headless' "
                     "for full functionality. Proceeding with basic torchvision transforms.")
    
    main_segmentation_pipeline()

