import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import pickle
from typing import Tuple, List, Dict, Optional
import warnings
from collections import Counter
import itertools

# Suppress specific warnings, e.g., from Albumentations or PyTorch
warnings.filterwarnings('ignore')

# Configure logging
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

class FetalCHDClassificationDataset(Dataset):
    """
    Dataset for fetal CHD classification.
    Supports loading images, and optionally, corresponding segmentation features.
    Handles annotations for classification labels and confidence scores.
    """
    
    def __init__(self, annotations_file, images_dir, transform=None, 
                 use_segmentation_features=False, segmentation_dir=None):
        """
        Initializes the FetalCHDClassificationDataset.

        Args:
            annotations_file (str or Path): Path to the JSON file containing image annotations.
            images_dir (str or Path): Directory containing the classification images.
            transform (albumentations.Compose or torchvision.transforms.Compose, optional):
                      Data augmentation and preprocessing transforms.
            use_segmentation_features (bool): If True, attempts to load segmentation masks
                                              as additional features.
            segmentation_dir (str or Path, optional): Directory containing predicted segmentation masks.
                                                      Required if `use_segmentation_features` is True.
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.use_segmentation_features = use_segmentation_features
        self.segmentation_dir = Path(segmentation_dir) if segmentation_dir else None
        
        # Load annotations from JSON file
        try:
            with open(annotations_file, 'r') as f:
                annotation_data = json.load(f)
            self.annotations = annotation_data.get('annotations', {})
        except FileNotFoundError:
            logger.error(f"Annotations file not found: {annotations_file}")
            self.annotations = {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from annotations file: {annotations_file}")
            self.annotations = {}
        
        # Define label mapping for classification classes
        self.label_map = {'normal': 0, 'asd': 1, 'vsd': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Process annotations and collect valid data samples
        self.data = []
        for image_id, annotation in self.annotations.items():
            image_path = self.images_dir / f"{image_id}.png" # Try .png first
            if not image_path.exists():
                # Try other common image extensions
                found_image = False
                for ext in ['.jpg', '.jpeg']:
                    temp_path = self.images_dir / f"{image_id}{ext}"
                    if temp_path.exists():
                        image_path = temp_path
                        found_image = True
                        break
                if not found_image:
                    logger.warning(f"Image not found for ID: {image_id} in {self.images_dir}. Skipping.")
                    continue
            
            # Ensure classification label exists and is valid
            classification_label_str = annotation.get('classification')
            if classification_label_str not in self.label_map:
                logger.warning(f"Invalid or missing classification label for {image_id}. Skipping.")
                continue
            
            label = self.label_map[classification_label_str]
            confidence = annotation.get('confidence', 0.0) # Default confidence if not present
            
            self.data.append({
                'image_path': image_path,
                'label': label,
                'confidence': confidence,
                'image_id': image_id
            })
        
        logger.info(f"Loaded {len(self.data)} annotated images for classification.")
        
        # Print class distribution for sanity check
        if self.data:
            label_counts = Counter([item['label'] for item in self.data])
            logger.info("Class Distribution:")
            for label_id, count in sorted(label_counts.items()):
                class_name = self.reverse_label_map.get(label_id, f"Unknown_{label_id}")
                logger.info(f"  {class_name}: {count} samples")
        else:
            logger.warning("No data samples loaded. Check annotations file and image directory.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a data sample (image, label, confidence, optional segmentation features).

        Returns:
            dict: A dictionary containing 'image', 'label', 'confidence', 'image_id',
                  and optionally 'segmentation' if `use_segmentation_features` is True.
        """
        item = self.data[idx]
        
        # Load image
        image = cv2.imread(str(item['image_path']))
        if image is None:
            logger.error(f"Failed to load image: {item['image_path']}")
            # Return dummy data or raise error to prevent pipeline breakage
            # For robustness, returning dummy data is safer in __getitem__
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            return {
                'image': ToTensorV2()(dummy_image) if A else transforms.ToTensor()(Image.fromarray(dummy_image)),
                'label': torch.tensor(0, dtype=torch.long),
                'confidence': torch.tensor(0.0, dtype=torch.float),
                'image_id': item['image_id']
            }
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB for consistency
        
        # Load segmentation features if enabled and path exists
        segmentation_features = None
        if self.use_segmentation_features and self.segmentation_dir:
            seg_path = self.segmentation_dir / f"{item['image_id']}_predicted_mask.png"
            if seg_path.exists():
                segmentation_features = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
                if segmentation_features is None:
                    logger.warning(f"Failed to load segmentation mask: {seg_path}")
            # else:
            #     logger.debug(f"Segmentation mask not found for {item['image_id']} at {seg_path}")
        
        # Apply transforms
        if self.transform:
            if A is not None: # Use Albumentations
                if segmentation_features is not None:
                    # Albumentations can transform image and mask together
                    transformed = self.transform(image=image, mask=segmentation_features)
                    image = transformed['image']
                    segmentation_features = transformed['mask']
                else:
                    transformed = self.transform(image=image)
                    image = transformed['image']
            else: # Fallback to torchvision
                # Torchvision transforms typically expect PIL Image or Tensor
                image = transforms.ToPILImage()(image)
                image = self.transform(image)
                # Segmentation features would need separate, manual resizing/conversion
                if segmentation_features is not None:
                    segmentation_features = torch.from_numpy(segmentation_features).float() / 255.0
        
        # Ensure image is a tensor (if not already by ToTensorV2)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 # HWC to CHW, normalize
        
        label = torch.tensor(item['label'], dtype=torch.long)
        confidence = torch.tensor(item['confidence'], dtype=torch.float)
        
        result = {
            'image': image,
            'label': label,
            'confidence': confidence,
            'image_id': item['image_id']
        }
        
        if segmentation_features is not None:
            if not isinstance(segmentation_features, torch.Tensor):
                # Ensure segmentation_features are float and potentially normalized
                segmentation_features = torch.from_numpy(segmentation_features).float() / 255.0
            result['segmentation'] = segmentation_features
        
        return result

class MultiModalCHDClassifier(nn.Module):
    """
    Advanced multi-modal classifier combining image and (optional) segmentation features.
    Uses a state-of-the-art backbone (e.g., EfficientNet) with attention mechanisms.
    Includes a classification head and an auxiliary confidence prediction head.
    """
    
    def __init__(self, model_name='efficientnet_b4', num_classes=3, 
                 use_segmentation=False, pretrained=True, dropout_rate=0.5):
        """
        Initializes the MultiModalCHDClassifier.

        Args:
            model_name (str): Name of the image backbone model (e.g., 'efficientnet_b4').
            num_classes (int): Number of classification output classes (e.g., 3 for Normal, ASD, VSD).
            use_segmentation (bool): If True, enables the segmentation feature branch.
            pretrained (bool): If True, loads pre-trained weights for backbones.
            dropout_rate (float): Dropout rate for the classification head.
        """
        super(MultiModalCHDClassifier, self).__init__()
        
        self.use_segmentation = use_segmentation
        self.num_classes = num_classes
        
        # Main image backbone (e.g., EfficientNet)
        # `num_classes=0` to remove the default classification head, `global_pool=''` to keep spatial features
        self.backbone = timm.create_model(model_name, pretrained=pretrained, 
                                        num_classes=0, global_pool='')
        backbone_features = self.backbone.num_features # Get output feature map channels
        
        # Segmentation processing branch (if enabled)
        if use_segmentation:
            # ResNet34 is a good choice for segmentation features
            self.seg_backbone = timm.create_model('resnet34', pretrained=pretrained,
                                                in_chans=1, num_classes=0, global_pool='') # in_chans=1 for grayscale mask
            seg_features = self.seg_backbone.num_features
            
            # Feature fusion module to combine image and segmentation features
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(backbone_features + seg_features, backbone_features, 
                         kernel_size=1, bias=False), # 1x1 conv to reduce channels after concat
                nn.BatchNorm2d(backbone_features),
                nn.ReLU(inplace=True)
            )
        
        # Global attention pooling to condense spatial features into a fixed-size vector
        self.attention_pool = AttentionPool2d(backbone_features, 256) # Output 256 features
        
        # Classification head with residual-like connections (BatchNorm + ReLU between layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), # BatchNorm after linear layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5), # Reduced dropout for deeper layers
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes) # Final layer for classification logits
        )
        
        # Auxiliary confidence prediction head
        # Predicts a scalar confidence score (e.g., 0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid() # Sigmoid to output a value between 0 and 1
        )
    
    def forward(self, x, segmentation=None):
        """
        Forward pass through the multi-modal classifier.

        Args:
            x (torch.Tensor): Input image tensor (main pathway).
            segmentation (torch.Tensor, optional): Input segmentation mask tensor (auxiliary pathway).

        Returns:
            tuple: (logits, confidence_prediction)
                   - logits (torch.Tensor): Raw classification scores.
                   - confidence_prediction (torch.Tensor): Predicted confidence score.
        """
        # Extract features from main image backbone
        features = self.backbone(x)
        
        if self.use_segmentation and segmentation is not None:
            # Expand segmentation to 4D (N, C, H, W) if it's 3D (N, H, W)
            if segmentation.ndim == 3:
                segmentation = segmentation.unsqueeze(1) # Add channel dimension
            
            # Extract features from segmentation backbone
            seg_features = self.seg_backbone(segmentation)
            
            # Resize segmentation features to match main features' spatial dimensions
            if seg_features.shape[2:] != features.shape[2:]:
                seg_features = F.interpolate(seg_features, size=features.shape[2:],
                                           mode='bilinear', align_corners=False)
            
            # Concatenate and fuse features
            fused_features = torch.cat([features, seg_features], dim=1)
            features = self.feature_fusion(fused_features)
        
        # Global attention pooling
        pooled_features = self.attention_pool(features)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        # Confidence prediction
        confidence = self.confidence_head(pooled_features)
        
        return logits, confidence

class AttentionPool2d(nn.Module):
    """
    Attention-based global pooling module.
    Learns an attention map to weight spatial features before global pooling,
    allowing the model to focus on important regions.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the AttentionPool2d module.

        Args:
            in_features (int): Number of input channels/features.
            out_features (int): Number of output features after pooling and projection.
        """
        super(AttentionPool2d, self).__init__()
        # Attention mechanism: 1x1 conv -> ReLU -> 1x1 conv -> Sigmoid
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, 1, kernel_size=1), # Output 1 channel for attention map
            nn.Sigmoid() # Normalize attention weights to 0-1
        )
        self.pool = nn.AdaptiveAvgPool2d(1) # Global average pooling
        # Optional linear projection if input and output feature dimensions differ
        self.fc = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through the AttentionPool2d.

        Args:
            x (torch.Tensor): Input feature map (N, C, H, W).

        Returns:
            torch.Tensor: Pooled and projected features (N, out_features).
        """
        # Generate attention map
        attention_map = self.attention(x)
        
        # Apply attention weighting to the input features
        weighted_features = x * attention_map
        
        # Global average pooling on the weighted features
        pooled = self.pool(weighted_features)
        pooled = pooled.view(pooled.size(0), -1) # Flatten to (N, C)
        
        # Project to the desired output dimension
        output = self.fc(pooled)
        
        return output

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification tasks.
    It down-weights easy examples and focuses on hard, misclassified examples.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initializes the FocalLoss.

        Args:
            alpha (float): Weighting factor for positive/negative examples (alpha or 1-alpha).
                           Can be a list/tensor for per-class alpha.
            gamma (float): Focusing parameter. Higher gamma reduces the loss for easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Computes the Focal Loss.

        Args:
            inputs (torch.Tensor): Raw, unnormalized scores (logits) from the model (N, C).
            targets (torch.Tensor): Ground truth class indices (N).

        Returns:
            torch.Tensor: Scalar Focal Loss.
        """
        # Compute standard Cross-Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Convert logits to probabilities
        pt = torch.exp(-ce_loss) # p_t = exp(-CE_loss)
        
        # Compute Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Combined loss for classification and confidence prediction.
    It combines a classification loss (e.g., Focal Loss) with an MSE loss for confidence.
    """
    
    def __init__(self, classification_weight=1.0, confidence_weight=0.1, 
                 use_focal_loss=True, focal_alpha=1.0, focal_gamma=2.0):
        """
        Initializes the MultiTaskLoss.

        Args:
            classification_weight (float): Weight for the classification loss.
            confidence_weight (float): Weight for the confidence prediction loss.
            use_focal_loss (bool): If True, uses Focal Loss for classification; otherwise, CrossEntropyLoss.
            focal_alpha (float): Alpha parameter for Focal Loss.
            focal_gamma (float): Gamma parameter for Focal Loss.
        """
        super(MultiTaskLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.confidence_weight = confidence_weight
        
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        self.confidence_loss = nn.MSELoss() # Mean Squared Error for regression-like confidence
    
    def forward(self, outputs, targets, ground_truth_confidence):
        """
        Computes the combined multi-task loss.

        Args:
            outputs (tuple): Tuple containing (classification_logits, predicted_confidence).
            targets (torch.Tensor): Ground truth class labels.
            ground_truth_confidence (torch.Tensor): Ground truth confidence scores (e.g., 1-5 scale).

        Returns:
            tuple: (total_loss, classification_loss, confidence_loss)
        """
        logits, predicted_confidence = outputs
        
        # Classification loss
        cls_loss = self.classification_loss(logits, targets)
        
        # Confidence loss (normalize ground truth confidence to 0-1 range if needed)
        # Assuming ground_truth_confidence is originally on a 1-5 scale, normalize to 0-1
        # (value - min) / (max - min) -> (conf - 1) / (5 - 1)
        normalized_confidence = (ground_truth_confidence - 1) / 4.0 
        conf_loss = self.confidence_loss(predicted_confidence.squeeze(), 
                                       normalized_confidence)
        
        total_loss = (self.classification_weight * cls_loss + 
                     self.confidence_weight * conf_loss)
        
        return total_loss, cls_loss, conf_loss

class CHDClassificationTrainer:
    """
    Comprehensive trainer for fetal CHD classification model.
    Includes training loop, validation, checkpointing, and metrics tracking.
    Supports class weighting and learning rate scheduling.
    """
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-4, weight_decay=1e-5, use_class_weights=True):
        """
        Initializes the CHDClassificationTrainer.

        Args:
            model (nn.Module): The classification model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to train on ('cuda' or 'cpu').
            learning_rate (float): Initial learning rate for the optimizer.
            weight_decay (float): L2 regularization parameter for the optimizer.
            use_class_weights (bool): If True, computes and applies class weights to the loss.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weights:
            # Get all labels from training data to compute class weights
            all_labels = []
            for batch in train_loader:
                all_labels.extend(batch['label'].cpu().numpy()) # Ensure labels are on CPU for numpy conversion
            
            # Compute balanced class weights
            class_weights = compute_class_weight('balanced', 
                                               classes=np.unique(all_labels),
                                               y=all_labels)
            class_weights = torch.FloatTensor(class_weights).to(device)
            logger.info(f"Using class weights: {class_weights}")
        
        # Optimizer with different learning rates for backbone and classifier head
        # This is a common practice for fine-tuning pre-trained models.
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name: # Identify parameters belonging to the pre-trained backbone
                backbone_params.append(param)
            else: # Parameters for the new classification head or fusion layers
                classifier_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': learning_rate} # Higher LR for new layers
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler: CosineAnnealingWarmRestarts for cyclical LR
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        # Loss function: Multi-task loss with optional Focal Loss for classification
        self.criterion = MultiTaskLoss(use_focal_loss=True, focal_alpha=1.0, focal_gamma=2.0)
        
        # Metrics tracking lists
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        # Best model tracking for checkpointing
        self.best_val_f1 = 0.0
        self.best_model_state = None
        
        # TensorBoard logging for visualization of training progress
        self.writer = SummaryWriter('runs/chd_classification')
    
    def train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            confidences = batch['confidence'].to(self.device)
            
            # Load segmentation features if available in the batch
            segmentation = batch.get('segmentation', None)
            if segmentation is not None:
                segmentation = segmentation.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad() # Zero gradients before backward pass
            outputs = self.model(images, segmentation) # Pass segmentation if available
            
            # Calculate multi-task loss
            loss, cls_loss, conf_loss = self.criterion(outputs, labels, confidences)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Gradient clipping
            self.optimizer.step()
            
            # Calculate accuracy for monitoring
            logits, _ = outputs
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar with real-time metrics
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.4f}',
                'Cls Loss': f'{cls_loss.item():.4f}',
                'Conf Loss': f'{conf_loss.item():.4f}'
            })
            
            # Log metrics to TensorBoard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Total_Loss', loss.item(), step)
            self.writer.add_scalar('Train/Classification_Loss', cls_loss.item(), step)
            self.writer.add_scalar('Train/Confidence_Loss', conf_loss.item(), step)
            self.writer.add_scalar('Train/Accuracy', accuracy, step)
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, epoch):
        """Validates the model on the validation set."""
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_confidences = []
        
        with torch.no_grad(): # Disable gradient calculation
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                confidences = batch['confidence'].to(self.device)
                
                segmentation = batch.get('segmentation', None)
                if segmentation is not None:
                    segmentation = segmentation.to(self.device)
                
                outputs = self.model(images, segmentation)
                loss, _, _ = self.criterion(outputs, labels, confidences)
                
                logits, confidence_pred = outputs
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidence_pred.cpu().numpy())
        
        # Calculate overall metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        # Generate classification report for per-class metrics
        report = classification_report(all_labels, all_predictions, 
                                     target_names=['Normal', 'ASD', 'VSD'], # Assuming these are the class names
                                     output_dict=True, zero_division=0)
        
        # Log validation metrics to TensorBoard
        self.writer.add_scalar('Val/Total_Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Val/F1_Score_Weighted', f1_weighted, epoch)
        
        return avg_loss, accuracy, f1_weighted, report, np.array(all_predictions), np.array(all_labels), np.array(all_probabilities), np.array(all_confidences)
    
    def train(self, num_epochs, save_dir='classification_checkpoints'):
        """Runs the complete training loop for the specified number of epochs."""
        os.makedirs(save_dir, exist_ok=True) # Create save directory if it doesn't exist
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate the model
            val_loss, val_acc, val_f1, val_report, val_preds, val_labels, val_probs, val_confs = self.validate(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1 (Weighted): {val_f1:.4f}")
            
            # Save the best model based on validation F1 score
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy() # Save a copy of the best state
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'classification_report': val_report
                }, os.path.join(save_dir, 'best_model.pth'))
                
                logger.info(f"New best model saved with F1: {val_f1:.4f}")
            
            # Save a checkpoint periodically (e.g., every 10 epochs)
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies,
                    'val_f1_scores': self.val_f1_scores
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                logger.info(f"Saved checkpoint for epoch {epoch+1}")
        
        logger.info(f"Training completed. Final best validation F1: {self.best_val_f1:.4f}")
        
        # Load the best model state back into the model at the end of training
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state for final evaluation/inference.")
        
        self.writer.close() # Close TensorBoard writer
    
    def plot_training_history(self, save_path='classification_training_history.png'):
        """
        Plots comprehensive training history including loss, accuracy, F1 score, and learning rate.

        Args:
            save_path (str): Path to save the generated plot.
        """
        if not self.train_losses or not self.val_losses or not self.val_accuracies or not self.val_f1_scores:
            logger.warning("No complete training history to plot. Run training first.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax3.plot(epochs, self.val_f1_scores, 'm-', label='Validation F1 Score (Weighted)', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Validation F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate plot
        # Extract LR from optimizer's param_groups (assuming first group for main LR)
        lr_history = [self.optimizer.param_groups[0]['lr'] for _ in epochs] # Simplified for constant LR or single group
        if hasattr(self.scheduler, 'get_last_lr'): # For schedulers that track LR
             lr_history = [self.scheduler.get_last_lr()[0] for _ in epochs] # This needs to be called after each scheduler step
             # A more accurate way would be to store LR at each step during training.
             # For plotting history after training, we can approximate.
             # Or, if using CosineAnnealingWarmRestarts, its LR is predictable.
        
        # To get actual LR history, you'd need to log it inside train_epoch or validate.
        # For now, if the scheduler is active, this will show a simplified view.
        # If the scheduler is CosineAnnealingWarmRestarts, the LR changes per batch,
        # so plotting epoch-wise LR requires averaging or tracking.
        # For simplicity, let's just plot the base LR or a dummy if not tracked.
        
        # A better way to plot LR: store self.optimizer.param_groups[0]['lr'] after each scheduler.step()
        # For now, let's use a dummy if not explicitly tracked.
        # This part of the plot might not be accurate without more complex LR tracking.
        # For a simple plot, we can just show the initial LR.
        
        # If you want to accurately plot LR, modify train() to store `self.optimizer.param_groups[0]['lr']`
        # at the end of each epoch after `self.scheduler.step()`.
        
        # Placeholder for LR history if not explicitly tracked per epoch
        # This part of the plot might not be accurate without more complex LR tracking.
        # For now, let's just plot the initial LR as a flat line or skip if not available.
        if len(self.optimizer.param_groups) > 0:
            initial_lr = self.optimizer.param_groups[0]['lr']
            lr_values_for_plot = [initial_lr] * len(epochs)
            # If a scheduler is used that changes LR frequently, this plot needs actual LR history.
            # For CosineAnnealingWarmRestarts, LR changes on each batch.
            # For simplicity in this script, we'll plot a placeholder.
            # To get actual LR history, you'd need to save `self.optimizer.param_groups[0]['lr']`
            # at the end of each epoch in `self.train()` and then plot that list.
            
            # A more robust way to get LR history for plotting:
            # During training, after `self.scheduler.step()`, append `self.optimizer.param_groups[0]['lr']`
            # to a `self.lr_history` list. Then plot that list here.
            
            # For now, just plot a dummy line to avoid errors if lr_history isn't populated
            if hasattr(self, 'lr_history') and len(self.lr_history) == len(epochs):
                 ax4.plot(epochs, self.lr_history, 'orange', linewidth=2)
            else:
                 # Fallback: simple line at initial LR
                 ax4.plot(epochs, [self.optimizer.param_groups[0]['lr']] * len(epochs), 'orange', linestyle=':', label='Initial LR')
                 logger.warning("Learning rate history not fully tracked. Plot shows initial LR or simplified view.")

            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.set_title('Learning Rate (N/A)')
            ax4.text(0.5, 0.5, 'No optimizer param groups found', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Classification training history plot saved to: {save_path}")
        plt.show()
        plt.close(fig)

class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations
    for classification tasks.
    """
    
    def __init__(self, model, test_loader, device, class_names=['Normal', 'ASD', 'VSD']):
        """
        Initializes the ModelEvaluator.

        Args:
            model (nn.Module): The trained classification model.
            test_loader (DataLoader): DataLoader for test/evaluation data.
            device (torch.device): Device to run evaluation on.
            class_names (list): List of class names corresponding to labels.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(self, save_dir='evaluation_results'):
        """
        Performs a complete model evaluation, calculating various metrics and
        generating visualization plots.

        Args:
            save_dir (str): Directory to save evaluation results and plots.

        Returns:
            dict: A dictionary containing all calculated metrics and raw predictions.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval() # Set model to evaluation mode
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_confidences = []
        
        logger.info("Starting model evaluation...")
        
        with torch.no_grad(): # Disable gradient calculation
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                segmentation = batch.get('segmentation', None)
                if segmentation is not None:
                    segmentation = segmentation.to(self.device)
                
                outputs = self.model(images, segmentation)
                logits, confidence = outputs
                
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        # Convert lists to numpy arrays for metric calculation
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        all_confidences = np.array(all_confidences)
        
        # Calculate core metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        
        # Classification report provides precision, recall, f1-score per class
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # ROC AUC (one-vs-rest for multiclass)
        auc_scores = {}
        if self.num_classes > 1: # ROC AUC requires at least two classes
            try:
                for i, class_name in enumerate(self.class_names):
                    binary_labels = (all_labels == i).astype(int)
                    binary_probs = all_probabilities[:, i]
                    # Check if there's more than one unique label in binary_labels
                    if len(np.unique(binary_labels)) > 1:
                        auc_scores[class_name] = roc_auc_score(binary_labels, binary_probs)
                    else:
                        logger.warning(f"Skipping AUC for class '{class_name}': Only one class present in labels.")
            except ValueError as e:
                logger.error(f"Error calculating ROC AUC: {e}")
                auc_scores = {} # Reset if error occurs
        else:
            logger.warning("ROC AUC not applicable for single-class classification.")

        # Print summary results
        logger.info(f"\n--- Classification Evaluation Results ---")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score (Weighted): {f1_weighted:.4f}")
        logger.info(f"  F1 Score (Macro): {f1_macro:.4f}")
        
        for class_name, auc in auc_scores.items():
            logger.info(f"  AUC {class_name}: {auc:.4f}")
        
        logger.info("\nClassification Report:\n" + json.dumps(report, indent=2))

        # Create visualizations
        self.plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        if auc_scores: # Only plot ROC if AUC scores were successfully calculated
            self.plot_roc_curves(all_labels, all_probabilities, save_path=os.path.join(save_dir, 'roc_curves.png'))
            self.plot_precision_recall_curves(all_labels, all_probabilities, save_path=os.path.join(save_dir, 'pr_curves.png'))
        self.plot_confidence_distribution(all_confidences, all_predictions, all_labels, 
                                        save_path=os.path.join(save_dir, 'confidence_distribution.png'))
        
        # Save detailed results to JSON
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'auc_scores': auc_scores,
            'classification_report': report,
            'confusion_matrix': cm.tolist(), # Convert numpy array to list for JSON serialization
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist(),
            'confidences': all_confidences.tolist()
        }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed evaluation results saved to: {os.path.join(save_dir, 'evaluation_results.json')}")
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plots the confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {save_path}")
    
    def plot_roc_curves(self, labels, probabilities, save_path):
        """Plots ROC curves for each class."""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            binary_labels = (labels == i).astype(int)
            binary_probs = probabilities[:, i]
            
            # Only plot if there are both positive and negative samples for the class
            if len(np.unique(binary_labels)) > 1:
                fpr, tpr, _ = roc_curve(binary_labels, binary_probs)
                auc = roc_auc_score(binary_labels, binary_probs)
                plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
            else:
                logger.warning(f"Skipping ROC curve for class '{class_name}': Only one class present in labels.")

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1) # Diagonal line for random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curves plot saved to: {save_path}")
    
    def plot_precision_recall_curves(self, labels, probabilities, save_path):
        """Plots Precision-Recall curves for each class."""
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            binary_labels = (labels == i).astype(int)
            binary_probs = probabilities[:, i]
            
            # Only plot if there are both positive and negative samples for the class
            if len(np.unique(binary_labels)) > 1:
                precision, recall, _ = precision_recall_curve(binary_labels, binary_probs)
                plt.plot(recall, precision, linewidth=2, label=f'{class_name}')
            else:
                logger.warning(f"Skipping PR curve for class '{class_name}': Only one class present in labels.")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Precision-Recall curves plot saved to: {save_path}")
    
    def plot_confidence_distribution(self, confidences, predictions, labels, save_path):
        """Plots confidence distribution by class and correctness, and calibration plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        axes[0, 0].hist(confidences, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Overall Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence by true class
        for i, class_name in enumerate(self.class_names):
            class_mask = (labels == i)
            class_confidences = confidences[class_mask]
            if len(class_confidences) > 0:
                axes[0, 1].hist(class_confidences, bins=20, alpha=0.7, label=class_name)
        
        axes[0, 1].set_title('Confidence Distribution by True Class')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Confidence for correct vs incorrect predictions
        correct_mask = (predictions == labels)
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        if len(correct_confidences) > 0:
            axes[1, 0].hist(correct_confidences, bins=20, alpha=0.7, color='green', label='Correct')
        if len(incorrect_confidences) > 0:
            axes[1, 0].hist(incorrect_confidences, bins=20, alpha=0.7, color='red', label='Incorrect')
        axes[1, 0].set_title('Confidence Distribution by Prediction Correctness')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Confidence calibration plot
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences_binned = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper) # Use >= and < for bins
            
            if np.sum(in_bin) > 0: # Check if there are samples in the bin
                accuracy_in_bin = correct_mask[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                accuracies.append(accuracy_in_bin)
                confidences_binned.append(avg_confidence_in_bin)
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        if confidences_binned: # Only plot if there's data for calibration
            axes[1, 1].plot(confidences_binned, accuracies, 'o-', linewidth=2, label='Model')
        axes[1, 1].set_title('Confidence Calibration')
        axes[1, 1].set_xlabel('Mean Confidence')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confidence distribution and calibration plot saved to: {save_path}")


def main_classification_pipeline():
    """
    Main pipeline for Fetal CHD classification.
    Supports training with cross-validation and evaluation.
    Configurable via command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Fetal CHD Classification Training and Evaluation Pipeline.")
    
    # General arguments
    parser.add_argument("--annotations_file", type=str, default="data/first_trimester/fetal_chd_annotations.json",
                        help="Path to the JSON file containing classification annotations. Default: data/first_trimester/fetal_chd_annotations.json.")
    parser.add_argument("--images_dir", type=str, default="data/first_trimester/images",
                        help="Root directory for classification images. Default: data/first_trimester/images.")
    parser.add_argument("--segmentation_dir", type=str, default="data/first_trimester/predicted_masks",
                        help="Directory for predicted segmentation masks (if use_segmentation is True). Default: data/first_trimester/predicted_masks.")
    parser.add_argument("--use_segmentation", action="store_true",
                        help="Flag to enable using segmentation features in the classifier.")
    parser.add_argument("--model_name", type=str, default="efficientnet_b4",
                        help="Name of the backbone model (e.g., 'efficientnet_b4', 'resnet34'). Default: efficientnet_b4.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classification output classes (e.g., 3 for Normal, ASD, VSD). Default: 3.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and validation. Default: 16.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for the optimizer. Default: 1e-4.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs. Default: 100.")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="Number of folds for Stratified K-Fold cross-validation. Default: 5.")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                        help="Dropout rate for the classification head. Default: 0.5.")
    parser.add_argument("--checkpoint_save_base_dir", type=str, default="classification_checkpoints",
                        help="Base directory to save model checkpoints for each fold. Default: classification_checkpoints.")
    parser.add_argument("--evaluation_results_base_dir", type=str, default="classification_evaluation_results",
                        help="Base directory to save evaluation results and plots for each fold. Default: classification_evaluation_results.")
    parser.add_argument("--pretrained_backbone", action="store_true",
                        help="Use pretrained weights for the backbone model.")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use balanced class weights during training.")

    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform, _ = create_classification_transforms() # TTA not used in main pipeline
    
    # Load full dataset
    full_dataset = FetalCHDClassificationDataset(
        annotations_file=args.annotations_file,
        images_dir=args.images_dir,
        segmentation_dir=args.segmentation_dir if args.use_segmentation else None,
        use_segmentation_features=args.use_segmentation,
        transform=None # Transform applied per subset
    )
    
    if not full_dataset.data:
        logger.error("No data loaded for classification. Exiting pipeline.")
        return

    # Stratified K-Fold Cross-Validation
    # Extract labels from the dataset for stratification
    labels = [item['label'] for item in full_dataset.data]
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_dataset)), labels)):
        logger.info(f"\n--- Training Fold {fold + 1}/{args.num_folds} ---")
        
        # Create fold-specific subsets
        # Important: Create new Subset instances for each fold to apply different transforms
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # Assign transforms to subsets
        # For subsets, you need to set the transform on the subset object itself or
        # create a custom dataset class that takes a list of indices and applies the transform.
        # A common pattern is to wrap Subset with a class that applies the transform.
        # For simplicity here, we'll assume `FetalCHDClassificationDataset` can take
        # a list of indices or that the transform is applied directly.
        # A more robust way is to pass the transform to the dataset constructor
        # and then create separate Dataset instances for train/val.
        
        # For current FetalCHDClassificationDataset, we need to create new dataset objects
        # or have a mechanism for subsets to get their own transforms.
        # Let's simplify by creating new Dataset instances for train/val subsets.
        
        # Create separate Dataset instances for train/val with their respective transforms
        # This requires re-loading data, which might be inefficient for very large datasets.
        # A more efficient way is to modify FetalCHDClassificationDataset to accept indices
        # and apply transforms based on a mode (train/val).
        
        # For demonstration, we'll pass the transforms directly to the DataLoader,
        # assuming the Dataset's __getitem__ handles it, or that the dataset is simple enough
        # that a single transform is set for the whole dataset, then subsets are taken.
        # The current FetalCHDClassificationDataset constructor takes a transform.
        # So we need to create new dataset instances for train/val, or modify the dataset
        # to apply transforms based on a mode.
        
        # Let's adjust by creating new temporary datasets for train/val within the loop
        # that use the correct transforms. This is a common workaround for `random_split`
        # or `Subset` not handling transforms directly.
        
        # To handle transforms properly with StratifiedKFold and Subsets:
        # The `FetalCHDClassificationDataset` needs to be initialized *without* a transform
        # for the `full_dataset`. Then, when creating `train_subset` and `val_subset`,
        # you'd typically create new `Dataset` objects that *wrap* these subsets and
        # apply the correct transforms.
        
        # For now, let's assume `FetalCHDClassificationDataset`'s `__getitem__`
        # is flexible enough to handle the transform being passed, and we'll
        # just ensure the Subset's underlying dataset's transform is set.
        # This is not ideal for `Subset` directly, but works if `full_dataset`
        # is designed to have its transform swapped.
        
        # Correct approach for transforms with Subset:
        # 1. Create full_dataset *without* transform.
        # 2. Create train_indices and val_indices.
        # 3. Create two *new* Dataset instances, one for train, one for val,
        #    each initialized with the appropriate transform and pointing to the
        #    same underlying data source (or a filtered list of paths based on indices).
        
        # Given the current FetalCHDClassificationDataset, the simplest way
        # is to make the dataset's `transform` attribute mutable and change it
        # for the `val_subset`'s underlying dataset. This is a bit of a hack
        # but often seen in quick implementations.
        
        # More robust: create a wrapper Dataset for subsets
        class SubsetWithTransform(Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform
            
            def __getitem__(self, idx):
                item = self.subset[idx] # Get item from original dataset via subset
                # Apply transform to image and segmentation (if present)
                image = item['image']
                segmentation = item.get('segmentation')

                if self.transform:
                    if A is not None:
                        if segmentation is not None and isinstance(segmentation, torch.Tensor):
                            # Convert back to numpy for Albumentations if it's already a tensor
                            image_np = (item['image'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            seg_np = (item['segmentation'].cpu().numpy() * 255).astype(np.uint8)
                            transformed = self.transform(image=image_np, mask=seg_np)
                            item['image'] = transformed['image']
                            item['segmentation'] = transformed['mask']
                        else:
                            image_np = (item['image'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            transformed = self.transform(image=image_np)
                            item['image'] = transformed['image']
                    else: # Torchvision fallback
                        # Torchvision transforms expect PIL Image, so convert if needed
                        if isinstance(image, torch.Tensor):
                            image = transforms.ToPILImage()(image)
                        item['image'] = self.transform(image)
                        # Segmentation would need manual handling here if it's not already a tensor
                        if segmentation is not None and not isinstance(segmentation, torch.Tensor):
                            item['segmentation'] = transforms.ToTensor()(Image.fromarray(segmentation))

                return item
            
            def __len__(self):
                return len(self.subset)

        train_dataset_fold = SubsetWithTransform(Subset(full_dataset, train_idx), train_transform)
        val_dataset_fold = SubsetWithTransform(Subset(full_dataset, val_idx), val_transform)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset_fold, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset_fold, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # Create model for the current fold
        model = MultiModalCHDClassifier(
            model_name=args.model_name,
            num_classes=args.num_classes,
            use_segmentation=args.use_segmentation,
            pretrained=args.pretrained_backbone,
            dropout_rate=args.dropout_rate
        )
        
        # Create trainer for the current fold
        trainer = CHDClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            use_class_weights=args.use_class_weights
        )
        
        # Train model for the current fold
        fold_checkpoint_dir = Path(args.checkpoint_save_base_dir) / f'fold_{fold+1}'
        trainer.train(
            num_epochs=args.num_epochs,
            save_dir=str(fold_checkpoint_dir)
        )
        
        # Evaluate on validation set for the current fold
        fold_eval_dir = Path(args.evaluation_results_base_dir) / f'fold_{fold+1}'
        evaluator = ModelEvaluator(trainer.model, val_loader, device, 
                                   class_names=[full_dataset.reverse_label_map[i] for i in sorted(full_dataset.reverse_label_map.keys())])
        fold_result = evaluator.evaluate(save_dir=str(fold_eval_dir))
        fold_results.append(fold_result)
        
        # Plot training history for the current fold
        trainer.plot_training_history(str(fold_checkpoint_dir / f'fold_{fold+1}_training_history.png'))
    
    # Calculate cross-validation statistics across all folds
    cv_accuracy = np.mean([result['accuracy'] for result in fold_results])
    cv_f1_weighted = np.mean([result['f1_weighted'] for result in fold_results])
    cv_f1_macro = np.mean([result['f1_macro'] for result in fold_results])
    cv_std_accuracy = np.std([result['accuracy'] for result in fold_results])
    cv_std_f1_weighted = np.std([result['f1_weighted'] for result in fold_results])
    cv_std_f1_macro = np.std([result['f1_macro'] for result in fold_results])
    
    logger.info(f"\n--- Cross-Validation Results Summary ({args.num_folds} Folds) ---")
    logger.info(f"Accuracy: {cv_accuracy:.4f} ± {cv_std_accuracy:.4f}")
    logger.info(f"F1 Score (Weighted): {cv_f1_weighted:.4f} ± {cv_std_f1_weighted:.4f}")
    logger.info(f"F1 Score (Macro): {cv_f1_macro:.4f} ± {cv_std_f1_macro:.4f}")
    
    # Save cross-validation results
    cv_results = {
        'cv_accuracy_mean': cv_accuracy,
        'cv_accuracy_std': cv_std_accuracy,
        'cv_f1_weighted_mean': cv_f1_weighted,
        'cv_f1_weighted_std': cv_std_f1_weighted,
        'cv_f1_macro_mean': cv_f1_macro,
        'cv_f1_macro_std': cv_std_f1_macro,
        'fold_results': fold_results
    }
    
    cv_results_path = Path(args.evaluation_results_base_dir) / 'cross_validation_summary.json'
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    logger.info(f"Cross-validation summary saved to: {cv_results_path}")
    
    logger.info("Classification pipeline completed successfully!")

if __name__ == "__main__":
    main_classification_pipeline()

