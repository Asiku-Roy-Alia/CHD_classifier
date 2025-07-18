import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import cv2
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime
import warnings
import gc # For garbage collection
import psutil # For memory usage tracking

# ReportLab for PDF generation (ensure it's installed: pip install reportlab)
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from io import BytesIO
    import base64
    REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("ReportLab not found. PDF report generation will be disabled. "
                   "Install with: pip install reportlab")
    REPORTLAB_AVAILABLE = False

# Attempt to import Albumentations, provide fallback if not available
try:
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    logger.warning("Albumentations not found. Using torchvision transforms as fallback for preprocessing. "
                   "Install with: pip install albumentations opencv-python-headless")
    ALBUMENTATIONS_AVAILABLE = False

# Import models from your models directory
# Ensure your models/segmentation_model.py and models/classification_model.py are correctly defined
try:
    from models.segmentation_model import EfficientAttentionUNet, CLASS_COLORMAP, ID_TO_CLASS, NUM_CLASSES as SEG_NUM_CLASSES
    from models.classification_model import MultiModalCHDClassifier
except ImportError as e:
    logger.error(f"Failed to import models. Ensure 'models' directory is in PYTHONPATH and contains "
                 f"segmentation_model.py and classification_model.py. Error: {e}")
    # Define dummy classes to allow script to run partially for demonstration
    class EfficientAttentionUNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); logger.error("Dummy EfficientAttentionUNet used.")
        def forward(self, x): return torch.zeros(x.shape[0], 12, x.shape[2], x.shape[3])
    class MultiModalCHDClassifier(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); logger.error("Dummy MultiModalCHDClassifier used.")
        def forward(self, x, seg=None): return torch.zeros(x.shape[0], 3), torch.zeros(x.shape[0], 1)
    CLASS_COLORMAP = {0: (0,0,0)}; ID_TO_CLASS = {0: "background"}; SEG_NUM_CLASSES = 1

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """
    Configuration for the entire inference pipeline.
    Defines paths to models, device, and various operational parameters.
    """
    segmentation_model_path: str
    classification_model_paths: List[str]  # List of paths for ensemble models
    device: str = 'cuda'
    batch_size: int = 1 # Batch processing handled by ClinicalInferenceEngine for simplicity
    confidence_threshold: float = 0.5 # Threshold for flagging low confidence predictions
    ensemble_method: str = 'soft_voting'  # 'soft_voting', 'hard_voting'
    tta_enabled: bool = True # Test-Time Augmentation
    output_format: str = 'json'  # 'json', 'pdf' (if ReportLab is installed)
    save_visualizations: bool = True # Save plots of results
    clinical_mode: bool = True # Generate detailed clinical reports

class ModelEnsemble:
    """
    Manages an ensemble of classification models for robust prediction.
    Supports different ensemble strategies (soft voting, hard voting).
    """
    
    def __init__(self, model_paths: List[str], device: str = 'cuda', 
                 ensemble_method: str = 'soft_voting'):
        """
        Initializes the ModelEnsemble by loading multiple classification models.

        Args:
            model_paths (List[str]): List of file paths to trained classification model checkpoints.
            device (str): Device to load models onto ('cuda' or 'cpu').
            ensemble_method (str): Strategy for combining predictions ('soft_voting' or 'hard_voting').
        """
        self.device = device
        self.ensemble_method = ensemble_method
        self.models = []
        self.model_weights = [] # Weights for soft voting, typically based on validation performance
        
        # Load all models
        for i, model_path in enumerate(model_paths):
            logger.info(f"Loading classification model {i+1}/{len(model_paths)}: {model_path}")
            
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Instantiate MultiModalCHDClassifier.
                # Assuming it was trained with segmentation features and efficientnet_b4.
                # Adjust these parameters if your classification models were trained differently.
                model = MultiModalCHDClassifier(
                    model_name='efficientnet_b4',
                    num_classes=3, # Normal, ASD, VSD
                    use_segmentation=True, # Assuming models were trained with segmentation features
                    pretrained=False # Already loaded from checkpoint, no need for new pretraining
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval() # Set model to evaluation mode
                
                self.models.append(model)
                
                # Use validation F1-score from checkpoint as model weight if available
                val_f1 = checkpoint.get('val_f1', 1.0) # Default to 1.0 if not found
                self.model_weights.append(val_f1)
            except Exception as e:
                logger.error(f"Error loading classification model {model_path}: {e}. Skipping this model.")
                # If a model fails to load, its weight should be zero or it should be skipped.
                # For simplicity, we'll just skip it and warn.
                continue
        
        if not self.models:
            raise ValueError("No classification models were successfully loaded for ensemble.")

        # Normalize weights so they sum to 1
        total_weight = sum(self.model_weights)
        if total_weight == 0: # Avoid division by zero if all weights are zero
            self.model_weights = [1.0 / len(self.models)] * len(self.models) # Equal weighting
            logger.warning("All model weights were zero, defaulting to equal weighting.")
        else:
            self.model_weights = [w / total_weight for w in self.model_weights]
        
        logger.info(f"Loaded {len(self.models)} classification models with normalized weights: {[f'{w:.3f}' for w in self.model_weights]}")
    
    def predict(self, images: torch.Tensor, segmentations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates ensemble predictions for a batch of images.

        Args:
            images (torch.Tensor): Batch of input images.
            segmentations (Optional[torch.Tensor]): Optional batch of segmentation masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ensemble_predictions (torch.Tensor): Final predicted class IDs (batch_size).
            - ensemble_probs (torch.Tensor): Final predicted probabilities (batch_size, num_classes).
            - uncertainty (torch.Tensor): Uncertainty score (e.g., variance of probabilities) (batch_size).
            - ensemble_confidence (torch.Tensor): Ensemble confidence score (batch_size).
        """
        all_predictions = [] # Stores argmax predictions from each model
        all_probabilities = [] # Stores softmax probabilities from each model
        all_confidences = [] # Stores confidence output from each model's confidence head
        
        with torch.no_grad():
            for model in self.models:
                logits, confidence = model(images, segmentations)
                
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_confidences.append(confidence)
        
        # Stack predictions from all models
        # [n_models, batch_size]
        stacked_predictions = torch.stack(all_predictions, dim=0)  
        # [n_models, batch_size, n_classes]
        stacked_probabilities = torch.stack(all_probabilities, dim=0)  
        # [n_models, batch_size, 1]
        stacked_confidences = torch.stack(all_confidences, dim=0)  
        
        # Ensemble predictions based on the chosen method
        if self.ensemble_method == 'soft_voting':
            # Weighted average of probabilities
            weights_tensor = torch.tensor(self.model_weights, device=self.device).view(-1, 1, 1)
            ensemble_probs = (stacked_probabilities * weights_tensor).sum(dim=0)
            ensemble_predictions = torch.argmax(ensemble_probs, dim=1)
            
        elif self.ensemble_method == 'hard_voting':
            # Majority voting: for each sample, find the most frequent prediction
            # This requires converting one-hot predictions to class IDs and then finding mode
            # A simpler way is to count votes for each class and pick the max
            num_classes = stacked_probabilities.shape[2]
            # Create a tensor to hold votes for each class for each sample
            votes = torch.zeros(stacked_probabilities.shape[1], num_classes, device=self.device)
            for i in range(stacked_predictions.shape[0]): # Iterate through models
                votes.scatter_add_(1, stacked_predictions[i].unsqueeze(1), torch.ones_like(stacked_predictions[i].unsqueeze(1)).float())
            
            ensemble_predictions = torch.argmax(votes, dim=1)
            # For hard voting, ensemble_probs can be a one-hot representation of the majority vote
            ensemble_probs = F.one_hot(ensemble_predictions, num_classes=num_classes).float()
            
        else:  # Fallback to soft voting if method is unrecognized
            logger.warning(f"Unknown ensemble method '{self.ensemble_method}'. Defaulting to soft voting.")
            weights_tensor = torch.tensor(self.model_weights, device=self.device).view(-1, 1, 1)
            ensemble_probs = (stacked_probabilities * weights_tensor).sum(dim=0)
            ensemble_predictions = torch.argmax(ensemble_probs, dim=1)
        
        # Calculate uncertainty (e.g., variance of probabilities across models)
        # Higher variance indicates higher disagreement among models, thus higher uncertainty.
        prob_variance = torch.var(stacked_probabilities, dim=0) # Variance per class per sample
        uncertainty = torch.mean(prob_variance, dim=1)  # Mean variance across classes for each sample
        
        # Ensemble confidence (weighted average of individual model confidences)
        weights_conf_tensor = torch.tensor(self.model_weights, device=self.device).view(-1, 1, 1)
        ensemble_confidence = (stacked_confidences * weights_conf_tensor).sum(dim=0).squeeze()
        
        return ensemble_predictions, ensemble_probs, uncertainty, ensemble_confidence

class TestTimeAugmentation:
    """
    Applies Test-Time Augmentation (TTA) to input images and segmentation masks.
    Generates multiple augmented versions of an input, which can then be
    passed through the model and their predictions ensembled for more robust results.
    """
    
    def __init__(self, n_augmentations: int = 5, classification_image_size: Tuple[int, int] = (224, 224)):
        """
        Initializes the TestTimeAugmentation module.

        Args:
            n_augmentations (int): Number of augmented versions to generate.
            classification_image_size (Tuple[int, int]): Target (height, width) for classification images.
        """
        self.n_augmentations = n_augmentations
        self.classification_image_size = classification_image_size
        self.transforms = self._create_tta_transforms()
        if not ALBUMENTATIONS_AVAILABLE:
            logger.warning("Albumentations not available, TTA will use basic torchvision transforms, "
                           "which might limit augmentation variety.")
    
    def _create_tta_transforms(self):
        """
        Creates a list of different augmentation transforms for TTA.
        Includes a base transform and several common augmentations.
        """
        if ALBUMENTATIONS_AVAILABLE:
            base_transform = A.Compose([
                A.Resize(*self.classification_image_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            transforms_list = [base_transform]  # Original image (no augmentation)
            
            # Add common augmentations if more than 1 augmentation is requested
            if self.n_augmentations > 1:
                transforms_list.extend([
                    A.Compose([
                        A.Resize(*self.classification_image_size, interpolation=cv2.INTER_AREA),
                        A.HorizontalFlip(p=1.0), # Horizontal flip
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ]),
                    A.Compose([
                        A.Resize(*self.classification_image_size, interpolation=cv2.INTER_AREA),
                        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), # Brightness/Contrast
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ]),
                    A.Compose([
                        A.Resize(*self.classification_image_size, interpolation=cv2.INTER_AREA),
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=1.0, border_mode=cv2.BORDER_CONSTANT), # Small geometric transform
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ]),
                    A.Compose([
                        A.Resize(*self.classification_image_size, interpolation=cv2.INTER_AREA),
                        A.GaussianBlur(blur_limit=(3,3), p=1.0), # Small blur
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ])
                ])
            
            return transforms_list[:self.n_augmentations] # Return only the requested number of augmentations
        else: # Fallback using torchvision
            # Torchvision doesn't have direct combined image+mask transforms for TTA
            # This will apply basic image transforms. Mask will need separate handling.
            base_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.classification_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # For simplicity, if Albumentations is not available, we'll only do the base transform for TTA.
            # Real TTA with torchvision would require more complex manual image manipulation.
            return [base_transform] * self.n_augmentations # Repeat base transform N times


    def apply(self, image: np.ndarray, segmentation: Optional[np.ndarray] = None) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Applies TTA transforms to the input image and optional segmentation mask.

        Args:
            image (np.ndarray): Input image (H, W, C).
            segmentation (Optional[np.ndarray]): Optional segmentation mask (H, W).

        Returns:
            Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
            - augmented_images (List[torch.Tensor]): List of augmented image tensors.
            - augmented_segmentations (List[Optional[torch.Tensor]]): List of augmented segmentation tensors.
        """
        augmented_images = []
        augmented_segmentations = []
        
        for transform in self.transforms:
            if ALBUMENTATIONS_AVAILABLE:
                if segmentation is not None:
                    result = transform(image=image, mask=segmentation)
                    augmented_images.append(result['image'])
                    augmented_segmentations.append(result['mask'])
                else:
                    result = transform(image=image)
                    augmented_images.append(result['image'])
                    augmented_segmentations.append(None)
            else: # Torchvision fallback
                # For torchvision, apply transform to image. Segmentation needs manual handling.
                img_pil = Image.fromarray(image)
                augmented_images.append(transform(img_pil))
                # For segmentation, if no Albumentations, it's complex to apply same geometric TTA.
                # For simplicity, if segmentation is provided and A is not, we'll just resize it
                # and assume no further geometric augmentation for the mask.
                if segmentation is not None:
                    seg_pil = Image.fromarray(segmentation)
                    seg_resized = transforms.Resize(self.classification_image_size, interpolation=transforms.InterpolationMode.NEAREST)(seg_pil)
                    augmented_segmentations.append(transforms.ToTensor()(seg_resized).squeeze(0)) # Squeeze channel dim for grayscale
                else:
                    augmented_segmentations.append(None)
        
        return augmented_images, augmented_segmentations

class ClinicalInferenceEngine:
    """
    The core engine for fetal CHD diagnosis, integrating all pipeline steps:
    image preprocessing, quality assessment, segmentation, classification (ensemble + TTA),
    and clinical report generation.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initializes the ClinicalInferenceEngine.

        Args:
            config (InferenceConfig): Configuration object for the inference pipeline.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Load segmentation model
        logger.info("Loading segmentation model...")
        self.segmentation_model = EfficientAttentionUNet(num_classes=SEG_NUM_CLASSES) # Use imported NUM_CLASSES
        try:
            seg_checkpoint = torch.load(config.segmentation_model_path, map_location=self.device)
            if 'model_state_dict' in seg_checkpoint:
                self.segmentation_model.load_state_dict(seg_checkpoint['model_state_dict'])
            else:
                self.segmentation_model.load_state_dict(seg_checkpoint)
            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()
            logger.info("Segmentation model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading segmentation model from {config.segmentation_model_path}: {e}")
            self.segmentation_model = None # Set to None if loading fails
        
        # Load classification ensemble
        logger.info("Loading classification ensemble...")
        try:
            self.classification_ensemble = ModelEnsemble(
                model_paths=config.classification_model_paths,
                device=config.device,
                ensemble_method=config.ensemble_method
            )
            logger.info("Classification ensemble loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading classification ensemble: {e}")
            self.classification_ensemble = None # Set to None if loading fails
        
        # Initialize TTA
        self.tta = None
        if config.tta_enabled:
            self.tta = TestTimeAugmentation(n_augmentations=5, classification_image_size=(224, 224)) # Default TTA augs
        
        # Class names for classification results (Normal, ASD, VSD)
        self.class_names = ['Normal', 'ASD', 'VSD']
        self.risk_levels = {0: 'Low', 1: 'High', 2: 'High'} # Mapping class ID to risk level
        
        # Performance tracking lists (for batch processing summary)
        self.inference_times = []
        self.memory_usage = []
        
        logger.info("Clinical inference engine initialized.")
        if self.segmentation_model is None or self.classification_ensemble is None:
            logger.error("One or more models failed to load. Inference capabilities may be limited.")

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Preprocesses an image (loads if path, converts to RGB) and extracts basic metadata.
        Also performs image quality assessment.

        Args:
            image_input (Union[str, Path, np.ndarray]): Path to the image file or a numpy array (H, W, C).

        Returns:
            Tuple[np.ndarray, Dict]:
            - image (np.ndarray): The loaded and converted RGB image (H, W, 3).
            - metadata (Dict): Dictionary containing image metadata and quality assessment.
        """
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            image = cv2.imread(str(image_path))
            if image is None:
                # Fallback to PIL for more robust image loading
                try:
                    from PIL import Image as PILImage
                    img_pil = PILImage.open(image_path)
                    image = np.array(img_pil)
                    if image.ndim == 2: # Grayscale to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.ndim == 3 and image.shape[2] == 4: # RGBA to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    elif image.ndim == 3 and image.shape[2] == 3: # RGB to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    logger.warning(f"Used PIL fallback to load image: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to load image {image_path} with both cv2 and PIL: {e}")
                    raise FileNotFoundError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB format
            metadata = {
                'source_path': str(image_path),
                'filename': image_path.name,
                'original_shape': image.shape
            }
        elif isinstance(image_input, np.ndarray):
            image = image_input
            if image.ndim == 2: # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[2] == 4: # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.ndim == 3 and image.shape[2] != 3:
                raise ValueError("Input numpy array must be 2D (grayscale) or 3D (H,W,3/4).")
            metadata = {
                'source_path': 'numpy_array_input',
                'filename': 'array_input',
                'original_shape': image.shape
            }
        else:
            raise TypeError("image_input must be a string path, Path object, or numpy array.")
        
        # Quality assessment
        quality_score = self._assess_image_quality(image)
        metadata['quality_score'] = quality_score
        metadata['quality_level'] = self._categorize_quality(quality_score)
        
        return image, metadata
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assesses image quality using a combination of metrics:
        sharpness (Laplacian variance), contrast (standard deviation), brightness, and SNR approximation.
        Returns a normalized quality score between 0 and 1.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image # Assume grayscale if not 3-channel RGB

        # Sharpness (Laplacian variance) - higher is better
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast (standard deviation of pixel intensities) - higher is better (within limits)
        contrast = np.std(gray)
        
        # Brightness (mean intensity) - optimal around 127.5 for 0-255 scale
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5 # Score is higher closer to 127.5
        
        # Signal-to-noise ratio approximation (contrast / noise)
        # Noise estimated as standard deviation of difference between blurred and original image
        noise_estimate = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        snr = contrast / (noise_estimate + 1e-6) if noise_estimate > 1e-6 else 0.0 # Avoid division by zero
        
        # Normalize individual scores to a 0-1 range (approximate scaling factors)
        # These scaling factors might need calibration based on your dataset characteristics.
        sharpness_norm = min(sharpness / 1000.0, 1.0) # Assuming max sharpness around 1000
        contrast_norm = min(contrast / 80.0, 1.0) # Assuming max contrast std dev around 80
        snr_norm = min(snr / 15.0, 1.0) # Assuming max SNR around 15
        
        # Combine metrics with arbitrary weights (can be tuned)
        quality_score = (sharpness_norm * 0.4 + contrast_norm * 0.3 + 
                         brightness_score * 0.2 + snr_norm * 0.1)
        
        return quality_score
    
    def _categorize_quality(self, quality_score: float) -> str:
        """Categorizes a normalized quality score into descriptive levels."""
        if quality_score >= 0.8:
            return 'Excellent'
        elif quality_score >= 0.6:
            return 'Good'
        elif quality_score >= 0.4:
            return 'Moderate'
        else:
            return 'Poor'
    
    def generate_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a segmentation mask and pixel-wise confidence map for the input image.

        Args:
            image (np.ndarray): Input RGB image (H, W, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - segmentation_mask (np.ndarray): Predicted segmentation mask (H, W) with class IDs.
            - confidence_map (np.ndarray): Pixel-wise confidence for the predicted class (H, W).
        """
        if self.segmentation_model is None:
            logger.error("Segmentation model not loaded. Cannot generate segmentation.")
            # Return dummy mask/confidence if model is unavailable
            dummy_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            dummy_confidence = np.zeros(image.shape[:2], dtype=np.float32)
            return dummy_mask, dummy_confidence

        # Preprocess for segmentation model
        seg_target_size = (256, 256) # Assuming segmentation model input size
        if ALBUMENTATIONS_AVAILABLE:
            transform = A.Compose([
                A.Resize(*seg_target_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        else: # Torchvision fallback
            from PIL import Image as PILImage
            image_pil = PILImage.fromarray(image)
            transform = transforms.Compose([
                transforms.Resize(seg_target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            segmentation_output = self.segmentation_model(image_tensor)
            # If deep supervision, take main output
            if isinstance(segmentation_output, tuple):
                segmentation_output = segmentation_output[0]

            segmentation_probs = F.softmax(segmentation_output, dim=1)
            segmentation_mask_tensor = torch.argmax(segmentation_probs, dim=1)
            
            # Get pixel-wise confidence for the predicted class
            confidence_map_tensor = torch.max(segmentation_probs, dim=1)[0]
        
        # Convert to numpy and resize back to original image dimensions
        original_h, original_w = image.shape[:2]
        segmentation_mask = segmentation_mask_tensor.squeeze().cpu().numpy()
        confidence_map = confidence_map_tensor.squeeze().cpu().numpy()
        
        # Resize mask using nearest neighbor to preserve class IDs
        segmentation_mask = cv2.resize(segmentation_mask.astype(np.uint8), 
                                     (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        # Resize confidence map using linear interpolation
        confidence_map = cv2.resize(confidence_map, 
                                  (original_w, original_h), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return segmentation_mask, confidence_map
    
    def classify_image(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Classifies the image using the ensemble model, optionally with TTA.

        Args:
            image (np.ndarray): Input RGB image (H, W, 3).
            segmentation (np.ndarray): Corresponding segmentation mask (H, W).

        Returns:
            Dict: Dictionary containing classification predictions, probabilities,
                  uncertainty, and confidence.
        """
        if self.classification_ensemble is None:
            logger.error("Classification ensemble not loaded. Cannot classify image.")
            return {
                'predictions': -1, # Indicate error
                'probabilities': np.zeros(len(self.class_names)),
                'uncertainty': 1.0, # Max uncertainty
                'confidence': 0.0 # Min confidence
            }

        augmented_images = []
        augmented_segmentations = []
        
        # Apply TTA or single transform
        if self.config.tta_enabled and self.tta is not None:
            augmented_images, augmented_segmentations = self.tta.apply(image, segmentation)
        else:
            # Apply a single transform (consistent with classification model's expected input)
            cls_target_size = (224, 224)
            if ALBUMENTATIONS_AVAILABLE:
                transform = A.Compose([
                    A.Resize(*cls_target_size, interpolation=cv2.INTER_AREA),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                transformed = transform(image=image, mask=segmentation)
                augmented_images = [transformed['image']]
                augmented_segmentations = [transformed['mask']]
            else: # Torchvision fallback
                from PIL import Image as PILImage
                image_pil = PILImage.fromarray(image)
                transform = transforms.Compose([
                    transforms.Resize(cls_target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                augmented_images = [transform(image_pil)]
                # For segmentation, manually resize and convert
                seg_pil = PILImage.fromarray(segmentation)
                seg_resized = transforms.Resize(cls_target_size, interpolation=transforms.InterpolationMode.NEAREST)(seg_pil)
                augmented_segmentations = [transforms.ToTensor()(seg_resized).squeeze(0)] # Squeeze channel dim

        # Collect predictions from all augmented versions
        all_tta_predictions = []
        all_tta_probabilities = []
        all_tta_uncertainties = []
        all_tta_confidences = []

        for aug_image, aug_seg in zip(augmented_images, augmented_segmentations):
            # Ensure tensors are on the correct device and have batch dimension
            aug_image_tensor = aug_image.unsqueeze(0).to(self.device)
            aug_seg_tensor = aug_seg.unsqueeze(0).to(self.device) if aug_seg is not None else None
            
            # Get ensemble predictions for this augmented version
            predictions, probabilities, uncertainties, confidences = self.classification_ensemble.predict(
                aug_image_tensor, aug_seg_tensor
            )
            
            all_tta_predictions.append(predictions.cpu().numpy())
            all_tta_probabilities.append(probabilities.cpu().numpy())
            all_tta_uncertainties.append(uncertainties.cpu().numpy())
            all_tta_confidences.append(confidences.cpu().numpy())
        
        # Average TTA results across all augmented versions
        # Note: predictions are class IDs, averaging them is not meaningful.
        # Average probabilities, uncertainties, and confidences.
        avg_probabilities = np.mean(np.array(all_tta_probabilities), axis=0) # Average over TTA dim
        avg_uncertainties = np.mean(np.array(all_tta_uncertainties), axis=0)
        avg_confidences = np.mean(np.array(all_tta_confidences), axis=0)
        
        # Final prediction is argmax of averaged probabilities
        final_prediction_id = np.argmax(avg_probabilities, axis=1)[0] # Take first (and only) item in batch
        
        return {
            'predictions': final_prediction_id,
            'probabilities': avg_probabilities[0].tolist(), # Convert to list for JSON
            'uncertainty': float(avg_uncertainties[0]),
            'confidence': float(avg_confidences[0])
        }
    
    def generate_clinical_report(self, image_path: str, results: Dict, 
                               metadata: Dict, save_path: Optional[str] = None) -> Dict:
        """
        Generates a comprehensive clinical report based on AI analysis results.
        Can save the report as JSON or PDF.

        Args:
            image_path (str): Path to the original input image.
            results (Dict): Dictionary containing classification results.
            metadata (Dict): Dictionary containing image metadata (e.g., quality).
            save_path (Optional[str]): Path to save the report (e.g., 'report.json' or 'report.pdf').

        Returns:
            Dict: The generated report dictionary.
        """
        if not self.config.clinical_mode:
            logger.info("Clinical report generation is disabled by configuration.")
            return {}

        prediction_id = results['predictions']
        probabilities = results['probabilities']
        uncertainty = results['uncertainty']
        confidence = results['confidence']
        
        class_name = self.class_names[prediction_id]
        risk_level = self.risk_levels[prediction_id]
        
        # Clinical interpretation based on diagnosis
        if prediction_id == 0:  # Normal
            interpretation = "No significant congenital heart defect detected. The fetal heart appears to have normal anatomy with intact atrial and ventricular septa."
            recommendation = "Continue routine prenatal care. Follow standard fetal monitoring protocols."
        elif prediction_id == 1:  # ASD (Atrial Septal Defect)
            interpretation = "Possible Atrial Septal Defect (ASD) detected. This indicates a potential opening in the wall between the heart's two upper chambers (atria). Further investigation is warranted."
            recommendation = "Recommend detailed fetal echocardiography by a pediatric cardiologist. Consider additional monitoring and postnatal cardiac evaluation."
        elif prediction_id == 2:  # VSD (Ventricular Septal Defect)
            interpretation = "Possible Ventricular Septal Defect (VSD) detected. This indicates a potential opening in the wall between the heart's two lower chambers (ventricles). Urgent specialist consultation is advised."
            recommendation = "Recommend immediate detailed fetal echocardiography by a pediatric cardiologist. Consider genetic counseling and specialized prenatal care."
        else:
            interpretation = "Diagnosis inconclusive or outside expected categories. Expert review required."
            recommendation = "Refer to a specialist for comprehensive evaluation."
        
        # Risk stratification based on confidence and uncertainty
        risk_modifier = ""
        if uncertainty > 0.3: # Arbitrary threshold for high uncertainty
            risk_modifier = "Note: High prediction uncertainty. Manual expert review strongly recommended."
        elif confidence < self.config.confidence_threshold:
            risk_modifier = f"Note: Prediction confidence ({confidence:.3f}) is below threshold ({self.config.confidence_threshold:.2f}). Consider additional imaging or expert review."
        else:
            risk_modifier = "Prediction confidence is acceptable for clinical screening."
        
        # Construct the report dictionary
        report = {
            'report_id': f"CHD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'patient_info': {
                'image_source': metadata.get('filename', 'N/A'),
                'image_quality': metadata.get('quality_level', 'N/A'),
                'quality_score': f"{metadata.get('quality_score', 0.0):.3f}"
            },
            'ai_analysis': {
                'primary_diagnosis': class_name,
                'confidence_score': f"{confidence:.3f}",
                'probability_distribution': {
                    self.class_names[0]: f"{probabilities[0]:.3f}",
                    self.class_names[1]: f"{probabilities[1]:.3f}",
                    self.class_names[2]: f"{probabilities[2]:.3f}"
                },
                'uncertainty_score': f"{uncertainty:.3f}",
                'risk_level': risk_level
            },
            'clinical_interpretation': interpretation,
            'recommendations': recommendation,
            'quality_assurance': risk_modifier,
            'disclaimer': "This AI-assisted analysis is for screening purposes only and must be interpreted by qualified medical professionals. It does not replace clinical judgment or detailed diagnostic imaging."
        }
        
        # Save report if a save_path is provided
        if save_path:
            if save_path.endswith('.pdf') and REPORTLAB_AVAILABLE:
                self._generate_pdf_report_file(report, save_path, image_path, results, metadata)
            else:
                try:
                    with open(save_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Clinical report saved to: {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save JSON report to {save_path}: {e}")
        
        return report
    
    def _generate_pdf_report_file(self, report: Dict, save_path: str, 
                                image_path: str, results: Dict, metadata: Dict):
        """Generates a PDF clinical report using ReportLab."""
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab is not installed. Cannot generate PDF report.")
            return

        doc = SimpleDocTemplate(save_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                   fontSize=24, spaceAfter=30, alignment=1)
        story.append(Paragraph("Fetal CHD AI Analysis Report", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Report Info Table
        info_data = [
            ['Report ID:', report['report_id']],
            ['Generated:', report['timestamp'].split('.')[0]], # Remove milliseconds
            ['Image Source:', report['patient_info']['image_source']],
            ['Image Quality:', report['patient_info']['image_quality']],
            ['Quality Score:', report['patient_info']['quality_score']]
        ]
        info_table = Table(info_data, colWidths=[2.5*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")), # Header background
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey)
        ]))
        story.append(Paragraph("Report Details", styles['Heading3']))
        story.append(info_table)
        story.append(Spacer(1, 0.2 * inch))

        # AI Analysis Results
        story.append(Paragraph("AI Analysis Results", styles['Heading3']))
        ai_data = [
            ['Primary Diagnosis:', report['ai_analysis']['primary_diagnosis']],
            ['Confidence Score:', report['ai_analysis']['confidence_score']],
            ['Risk Level:', report['ai_analysis']['risk_level']],
            ['Uncertainty Score:', report['ai_analysis']['uncertainty_score']],
            ['Probabilities (Normal/ASD/VSD):', f"{report['ai_analysis']['probability_distribution']['Normal']} / {report['ai_analysis']['probability_distribution']['ASD']} / {report['ai_analysis']['probability_distribution']['VSD']}"],
        ]
        ai_table = Table(ai_data, colWidths=[2.5*inch, 4.5*inch])
        ai_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0EFFF")), # Header background
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey)
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 0.2 * inch))
        
        # Clinical Interpretation
        story.append(Paragraph("Clinical Interpretation", styles['Heading3']))
        story.append(Paragraph(report['clinical_interpretation'], styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading3']))
        story.append(Paragraph(report['recommendations'], styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
        
        # Quality Assurance
        story.append(Paragraph("Quality Assurance", styles['Heading3']))
        story.append(Paragraph(report['quality_assurance'], styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        # Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
                                        fontSize=9, textColor=colors.red, spaceBefore=10)
        story.append(Paragraph("Disclaimer", styles['Heading3']))
        story.append(Paragraph(report['disclaimer'], disclaimer_style))
        story.append(Spacer(1, 0.2 * inch))

        # Add visualizations if they exist
        visualization_path = Path(save_path).parent / f"{Path(image_path).stem}_visualization.png"
        if visualization_path.exists():
            try:
                img = Image(str(visualization_path), width=400, height=300) # Adjust size as needed
                story.append(Paragraph("Visual Analysis", styles['Heading3']))
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))
            except Exception as e:
                logger.warning(f"Could not embed visualization image {visualization_path} in PDF: {e}")
        
        try:
            doc.build(story)
            logger.info(f"PDF report saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error building PDF report {save_path}: {e}")
    
    def _save_visualizations(self, image: np.ndarray, segmentation: np.ndarray, 
                           confidence: np.ndarray, classification_results: Dict,
                           output_dir: str, filename: str):
        """
        Saves visualization plots for a single processed image.
        Plots original image, segmentation mask, segmentation confidence, and classification probabilities.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Segmentation mask
        # Create a colored mask using the defined colormap
        colored_seg_mask = np.zeros((*segmentation.shape, 3), dtype=np.float32)
        for class_id, color in CLASS_COLORMAP.items():
            colored_seg_mask[segmentation == class_id] = color
        axes[0, 1].imshow(colored_seg_mask)
        axes[0, 1].set_title('Segmentation Mask')
        axes[0, 1].axis('off')
        
        # Add a legend for segmentation classes
        handles = [plt.Line2D([0], [0], color=color, lw=4, label=ID_TO_CLASS[id]) 
                   for id, color in CLASS_COLORMAP.items() if id < SEG_NUM_CLASSES] # Ensure valid IDs
        axes[0, 1].legend(handles=handles, title="Seg. Classes", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        # 3. Segmentation confidence map
        im = axes[1, 0].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title('Segmentation Confidence')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
        
        # 4. Classification probabilities
        probabilities = classification_results['probabilities']
        class_names = self.class_names
        colors = ['green', 'orange', 'red'] # Colors for Normal, ASD, VSD
        
        bars = axes[1, 1].bar(class_names, probabilities, color=colors, alpha=0.7)
        axes[1, 1].set_title('Classification Probabilities')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for segmentation legend
        save_path = Path(output_dir) / f"{filename}_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualizations saved to: {save_path}")
    
    def process_single_image(self, image_path: Union[str, Path], output_dir: Optional[str] = None) -> Dict:
        """
        Processes a single image through the entire pipeline: preprocess, segment, classify, report.

        Args:
            image_path (Union[str, Path]): Path to the input image.
            output_dir (Optional[str]): Directory to save results (report, visualizations).

        Returns:
            Dict: A dictionary containing all results and performance metrics.
        """
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024) # MB
        
        try:
            # 1. Preprocess image and assess quality
            image, metadata = self.preprocess_image(image_path)
            
            # 2. Generate segmentation mask and confidence
            segmentation_mask, segmentation_confidence = self.generate_segmentation(image)
            
            # 3. Classify image using ensemble and TTA
            classification_results = self.classify_image(image, segmentation_mask)
            
            # Prepare output paths
            output_dir_path = Path(output_dir) if output_dir else None
            if output_dir_path:
                output_dir_path.mkdir(parents=True, exist_ok=True)
                report_json_path = output_dir_path / f"{Path(image_path).stem}_report.json"
                report_pdf_path = output_dir_path / f"{Path(image_path).stem}_report.pdf"
            else:
                report_json_path = None
                report_pdf_path = None
            
            # 4. Generate clinical report (JSON and/or PDF)
            clinical_report = self.generate_clinical_report(
                str(image_path), classification_results, metadata, 
                str(report_json_path) if report_json_path else None # Save JSON report
            )
            
            # 5. Generate visualizations if requested
            if self.config.save_visualizations and output_dir_path:
                self._save_visualizations(image, segmentation_mask, segmentation_confidence,
                                          classification_results, str(output_dir_path), Path(image_path).stem)
            
            # 6. Generate PDF report if clinical mode is enabled and ReportLab is available
            if self.config.clinical_mode and REPORTLAB_AVAILABLE and report_pdf_path:
                self._generate_pdf_report_file(clinical_report, str(report_pdf_path), str(image_path), 
                                                classification_results, metadata)

            # Calculate performance metrics
            processing_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_used = final_memory - initial_memory
            
            self.inference_times.append(processing_time)
            self.memory_usage.append(memory_used)
            
            # Compile final results for return
            final_results = {
                'image_path': str(image_path),
                'metadata': metadata,
                'segmentation_summary': {
                    'mask_shape': segmentation_mask.shape,
                    'average_confidence': float(np.mean(segmentation_confidence)),
                    'min_confidence': float(np.min(segmentation_confidence)),
                    'max_confidence': float(np.max(segmentation_confidence))
                },
                'classification': classification_results,
                'clinical_report_summary': clinical_report, # Summary of the report
                'performance': {
                    'processing_time_seconds': processing_time,
                    'memory_used_mb': memory_used
                }
            }
            
            logger.info(f"Successfully processed {image_path.name} in {processing_time:.2f} seconds.")
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
        finally:
            # Clean up GPU memory and Python garbage collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def process_batch(self, image_dir: str, output_dir: str, 
                     file_extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> Dict:
        """
        Processes a batch of images from a directory through the complete pipeline.

        Args:
            image_dir (str): Directory containing input images.
            output_dir (str): Directory to save all results (reports, visualizations).
            file_extensions (List[str]): List of image file extensions to search for.

        Returns:
            Dict: A summary of the batch processing, including statistics and failed images.
        """
        image_dir_path = Path(image_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        
        # Find all images with specified extensions (case-insensitive)
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(list(image_dir_path.rglob(f"*{ext.lower()}")))
            image_paths.extend(list(image_dir_path.rglob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_paths)} images to process in batch mode.")
        
        results = [] # Stores successful processing results
        failed_images = [] # Stores information about failed images
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
            
            # Process each image individually
            result = self.process_single_image(str(image_path), str(output_dir_path))
            
            if 'error' in result:
                failed_images.append({'path': str(image_path), 'error': result['error']})
            else:
                results.append(result)
        
        # Generate and save batch summary
        batch_summary = self._generate_batch_summary(results, failed_images, output_dir_path)
        
        logger.info(f"Batch processing completed. {len(results)} successful, {len(failed_images)} failed.")
        
        return batch_summary
    
    def _generate_batch_summary(self, results: List[Dict], failed_images: List[Dict], 
                              output_dir: Path) -> Dict:
        """
        Generates a summary dictionary and plots for batch processing results.
        """
        total_images = len(results) + len(failed_images)
        if total_images == 0:
            return {'error': 'No images were found or processed.'}
        
        # Extract statistics from successful results
        processing_times = [r['performance']['processing_time_seconds'] for r in results]
        memory_usage = [r['performance']['memory_used_mb'] for r in results]
        
        predictions = [r['classification']['predictions'] for r in results]
        confidences = [r['classification']['confidence'] for r in results]
        uncertainties = [r['classification']['uncertainty'] for r in results]
        
        # Count classifications
        class_counts = {name: 0 for name in self.class_names}
        for pred_id in predictions:
            if pred_id in self.risk_levels: # Ensure valid prediction ID
                class_counts[self.class_names[pred_id]] += 1
        
        # Count quality levels
        quality_levels = [r['metadata']['quality_level'] for r in results]
        quality_counts = Counter(quality_levels)
        
        summary = {
            'batch_info': {
                'total_images': total_images,
                'successful_images': len(results),
                'failed_images_count': len(failed_images),
                'success_rate': len(results) / total_images if total_images > 0 else 0.0
            },
            'performance_metrics': {
                'avg_processing_time_seconds': np.mean(processing_times) if processing_times else 0.0,
                'std_processing_time_seconds': np.std(processing_times) if processing_times else 0.0,
                'total_processing_time_seconds': np.sum(processing_times) if processing_times else 0.0,
                'avg_memory_used_mb': np.mean(memory_usage) if memory_usage else 0.0,
                'max_memory_used_mb': np.max(memory_usage) if memory_usage else 0.0
            },
            'classification_summary': {
                'class_distribution': class_counts,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'std_confidence': np.std(confidences) if confidences else 0.0,
                'avg_uncertainty': np.mean(uncertainties) if uncertainties else 0.0,
                'std_uncertainty': np.std(uncertainties) if uncertainties else 0.0,
                'high_risk_cases_count': class_counts.get('ASD', 0) + class_counts.get('VSD', 0)
            },
            'quality_summary': {
                'avg_quality_score': np.mean([r['metadata']['quality_score'] for r in results]) if results else 0.0,
                'std_quality_score': np.std([r['metadata']['quality_score'] for r in results]) if results else 0.0,
                'quality_distribution': dict(quality_counts)
            },
            'failed_images_details': [{'path': f['path'], 'error': f['error']} for f in failed_images]
        }
        
        # Save summary to JSON
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str) # default=str to handle numpy types
        logger.info(f"Batch summary saved to: {summary_path}")
        
        # Generate summary visualization
        self._plot_batch_summary(summary, output_dir)
        
        return summary
    
    def _plot_batch_summary(self, summary: Dict, output_dir: Path):
        """Generates visualization plots for the batch processing summary."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Class distribution (Pie Chart)
        class_dist = summary['classification_summary']['class_distribution']
        if sum(class_dist.values()) > 0:
            axes[0, 0].pie(class_dist.values(), labels=class_dist.keys(), autopct='%1.1f%%',
                          colors=['green', 'orange', 'red'], startangle=90)
        axes[0, 0].set_title('Diagnosis Distribution')
        
        # 2. Image Quality Distribution (Bar Chart)
        quality_dist = summary['quality_summary']['quality_distribution']
        if quality_dist:
            # Order quality levels for consistent plotting
            ordered_quality_levels = ['Poor', 'Moderate', 'Good', 'Excellent']
            counts = [quality_dist.get(level, 0) for level in ordered_quality_levels]
            colors_map = {'Poor': 'red', 'Moderate': 'orange', 'Good': 'yellowgreen', 'Excellent': 'green'}
            bar_colors = [colors_map[level] for level in ordered_quality_levels]

            axes[0, 1].bar(ordered_quality_levels, counts, color=bar_colors)
        axes[0, 1].set_title('Image Quality Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Processing Time Distribution (Histogram)
        if hasattr(self, 'inference_times') and self.inference_times:
            axes[0, 2].hist(self.inference_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 2].set_title('Processing Time Distribution')
            axes[0, 2].set_xlabel('Time (seconds)')
            axes[0, 2].set_ylabel('Frequency')
        
        # 4. Performance Metrics (Bar Chart)
        perf_metrics_labels = ['Avg Proc. Time (s)', 'Avg Memory (MB)', 'Success Rate']
        perf_values = [
            summary['performance_metrics']['avg_processing_time_seconds'],
            summary['performance_metrics']['avg_memory_used_mb'],
            summary['batch_info']['success_rate']
        ]
        
        bars = axes[1, 0].bar(perf_metrics_labels, perf_values, color=['blue', 'green', 'purple'])
        axes[1, 0].set_title('Performance Summary')
        for bar, value in zip(bars, perf_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # 5. Risk Level Distribution (Pie Chart)
        high_risk = summary['classification_summary']['high_risk_cases_count']
        normal_cases = summary['classification_summary']['class_distribution'].get('Normal', 0)
        
        risk_data = ['Normal', 'High Risk']
        risk_counts = [normal_cases, high_risk]
        if sum(risk_counts) > 0:
            axes[1, 1].pie(risk_counts, labels=risk_data, autopct='%1.1f%%',
                          colors=['lightgreen', 'salmon'], startangle=90)
        axes[1, 1].set_title('Risk Level Distribution')
        
        # 6. Summary Statistics Table (Text)
        stats_data = [
            ['Total Images:', str(summary['batch_info']['total_images'])],
            ['Successful:', str(summary['batch_info']['successful_images'])],
            ['Failed:', str(summary['batch_info']['failed_images_count'])],
            ['Success Rate:', f"{summary['batch_info']['success_rate']:.1%}"],
            ['Avg Confidence:', f"{summary['classification_summary']['avg_confidence']:.3f}"],
            ['Avg Uncertainty:', f"{summary['classification_summary']['avg_uncertainty']:.3f}"],
            ['Avg Quality Score:', f"{summary['quality_summary']['avg_quality_score']:.3f}"],
            ['High Risk Cases:', str(summary['classification_summary']['high_risk_cases_count'])]
        ]
        
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table = axes[1, 2].table(cellText=stats_data, colWidths=[0.5, 0.5], cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Key Summary Statistics')
        
        plt.tight_layout()
        save_path = output_dir / 'batch_summary_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Batch summary visualization saved to: {save_path}")

def main():
    """Main function for command-line interface to run the inference system."""
    parser = argparse.ArgumentParser(description='Fetal CHD Clinical Inference System')
    
    # Required arguments
    parser.add_argument('--segmentation_model', type=str, required=True, 
                       help='Path to the trained segmentation model checkpoint (e.g., segmentation_checkpoints/best_model.pth).')
    parser.add_argument('--classification_models', nargs='+', required=True,
                       help='Paths to one or more trained classification model checkpoints for ensemble (e.g., classification_checkpoints/fold_1/best_model.pth).')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path: either a single image file (e.g., image.png) or a directory for batch processing.')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory where results (reports, visualizations) will be saved.')
    
    # Optional arguments with defaults
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference (e.g., "cuda" or "cpu"). Default: cuda.')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Confidence threshold for flagging predictions in clinical reports. Default: 0.6.')
    parser.add_argument('--ensemble_method', type=str, default='soft_voting',
                       choices=['soft_voting', 'hard_voting'],
                       help='Ensemble method for classification models. Default: soft_voting.')
    parser.add_argument('--no_tta', action='store_false', dest='tta_enabled',
                       help='Disable test-time augmentation (TTA). By default, TTA is enabled.')
    parser.add_argument('--no_clinical_mode', action='store_false', dest='clinical_mode',
                       help='Disable generation of detailed clinical reports. By default, clinical reports are enabled.')
    parser.add_argument('--no_visualizations', action='store_false', dest='save_visualizations',
                       help='Disable saving of visualization plots. By default, visualizations are saved.')
    
    args = parser.parse_args()
    
    # Create configuration object from parsed arguments
    config = InferenceConfig(
        segmentation_model_path=args.segmentation_model,
        classification_model_paths=args.classification_models,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        ensemble_method=args.ensemble_method,
        tta_enabled=args.tta_enabled,
        clinical_mode=args.clinical_mode,
        save_visualizations=args.save_visualizations
    )
    
    # Initialize the clinical inference engine
    engine = ClinicalInferenceEngine(config)
    
    # Determine if input is a file or a directory
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    if input_path.is_file():
        # Process a single image
        logger.info(f"Initiating single image processing for: {input_path.name}")
        result = engine.process_single_image(str(input_path), str(output_path))
        
        if 'error' not in result:
            logger.info("Single image processing completed successfully.")
            diagnosis = engine.class_names[result['classification']['predictions']]
            confidence = result['classification']['confidence']
            print(f"\n--- AI Diagnosis Summary ---")
            print(f"  Diagnosis: {diagnosis}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Report & Visualizations saved to: {output_path}")
        else:
            logger.error(f"Single image processing failed for {input_path.name}: {result['error']}")
            print(f"\n--- Processing Failed ---")
            print(f"  Image: {input_path.name}")
            print(f"  Error: {result['error']}")
    
    elif input_path.is_dir():
        # Process a batch of images
        logger.info(f"Initiating batch processing for directory: {input_path}")
        summary = engine.process_batch(str(input_path), str(output_path))
        
        logger.info("Batch processing completed.")
        if 'error' not in summary:
            print(f"\n--- Batch Processing Summary ---")
            print(f"  Total Images Processed: {summary['batch_info']['total_images']}")
            print(f"  Successful Images: {summary['batch_info']['successful_images']}")
            print(f"  Failed Images: {summary['batch_info']['failed_images_count']}")
            print(f"  Success Rate: {summary['batch_info']['success_rate']:.1%}")
            print(f"  High Risk Cases Detected: {summary['classification_summary']['high_risk_cases_count']}")
            print(f"  Detailed summary and visualizations saved to: {output_path}")
        else:
            logger.error(f"Batch processing failed: {summary['error']}")
            print(f"\n--- Batch Processing Failed ---")
            print(f"  Error: {summary['error']}")
    
    else:
        logger.error(f"Invalid input path type. Must be a file or a directory: {input_path}")

if __name__ == "__main__":
    main()

