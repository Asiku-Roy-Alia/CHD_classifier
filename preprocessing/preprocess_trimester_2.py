import argparse
import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pathlib import Path
from scipy import ndimage
from skimage import measure, morphology
import logging

# Configure logging for preprocessing script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltrasoundPreprocessorTrimester2:
    """
    Robust ROI extraction and preprocessing for fetal echocardiography frames for Trimester 2 (segmentation model).
    This script focuses on detecting and cleaning the ultrasound fan/sector.
    Instead of cropping, regions outside the detected fan are masked (filled with zeros/black),
    preserving the original image dimensions. Text and scale bar regions are inpainted.
    """
    def __init__(self,
                 erosion_kernel_size=5,
                 dilation_kernel_size=5,
                 text_inpaint_method='inpaint_telea', # 'inpaint_telea' or 'mask_fill'
                 padding_color=(0,0,0)):
        """
        Initializes the preprocessor with parameters for ROI extraction and text inpainting.

        Args:
            erosion_kernel_size (int): Size of the kernel for erosion in fan detection.
            dilation_kernel_size (int): Size of the kernel for dilation in fan detection.
            text_inpaint_method (str): Method for text inpainting ('inpaint_telea' or 'mask_fill').
            padding_color (tuple): Color to fill masked regions outside the fan.
        """
        self.erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        self.dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        self.text_inpaint_method = text_inpaint_method
        self.padding_color = padding_color


    def preprocess_image_base(self, image):
        """
        Converts image to uint8 and ensures it's BGR. Returns the BGR image and its grayscale version.
        Handles different input image data types and dimensions.
        """
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.ndim == 2:
            gray = image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray

    def detect_text_regions(self, image):
        """
        Detects horizontal text-like regions in the image using thresholding and morphological operations.
        Filters by aspect ratio and relative height to identify typical text blocks.
        Returns a mask for detected text regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1)) # Horizontal kernel
        text_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kh, iterations=2)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kh, iterations=3)

        # Invert the mask to find white text on black background
        inverted_binary = cv2.bitwise_not(binary)
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) # Vertical kernel
        inverted_text_mask = cv2.morphologyEx(inverted_binary, cv2.MORPH_OPEN, kv, iterations=2)
        inverted_text_mask = cv2.morphologyEx(inverted_text_mask, cv2.MORPH_CLOSE, kv, iterations=3)

        combined_text_mask = cv2.bitwise_or(text_mask, inverted_text_mask)
        return combined_text_mask

    def inpaint_text(self, image, text_mask):
        """
        Inpaints (removes) text regions from the image based on the detected text mask.
        Supports cv2.INPAINT_TELEA or simply filling with padding_color.
        """
        if self.text_inpaint_method == 'inpaint_telea':
            return cv2.inpaint(image, text_mask, 3, cv2.INPAINT_TELEA)
        elif self.text_inpaint_method == 'mask_fill':
            inpainted_image = image.copy()
            inpainted_image[text_mask == 255] = self.padding_color
            return inpainted_image
        return image # Fallback

    def find_fan_contour(self, gray_image):
        """
        Finds the largest bright region (ultrasound fan) in the grayscale image.
        Returns the contour and its mask.
        """
        # Threshold to isolate bright regions (ultrasound fan)
        _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
        # Apply morphological operations to clean up noise and consolidate regions
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.erosion_kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.dilation_kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the largest contour, which is likely the fan
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the largest contour
        fan_mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.drawContours(fan_mask, [largest_contour], -1, 255, cv2.FILLED)
        
        return largest_contour, fan_mask

    def preprocess(self, image_path: Path):
        """
        Performs the complete preprocessing pipeline for Trimester 2 images.
        Reads image, converts to BGR/grayscale, detects text, finds fan ROI,
        inpaints text, and masks regions outside the fan.

        Args:
            image_path (Path): Path to the input image file.

        Returns:
            tuple: A tuple containing the processed image (numpy array) and a
                   dictionary of metadata (original dimensions, fan contour info).
                   Returns (None, None) if processing fails.
        """
        try:
            # Handle DICOM files specifically
            if image_path.suffix.lower() == '.dcm':
                dicom_data = pydicom.dcmread(image_path)
                image = dicom_data.pixel_array
                # Convert DICOM image to uint8 and handle potential grayscale/color issues
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"Could not read image: {image_path}. Skipping.")
                return None, None

            orig_h, orig_w = image.shape[:2]
            processed_image, gray_image = self.preprocess_image_base(image.copy())

            # Step 1: Detect and inpaint text regions
            text_mask = self.detect_text_regions(processed_image)
            inpainted_image = self.inpaint_text(processed_image, text_mask)

            # Step 2: Find the ultrasound fan contour and create its mask
            fan_contour, fan_mask = self.find_fan_contour(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY))
            
            process_meta = {'has_fan_contour': False}
            if fan_mask is not None:
                process_meta['has_fan_contour'] = True
                
                # Step 3: Mask regions outside the detected fan
                # Fill outside the fan with padding_color (black)
                final_image = np.full(image.shape, self.padding_color, dtype=np.uint8)
                final_image[fan_mask == 255] = inpainted_image[fan_mask == 255]
            else:
                # If no fan contour found, just return the text-inpainted image
                logger.warning(f"No clear fan contour found for {image_path}. Returning text-inpainted image.")
                final_image = inpainted_image

            meta = {
                'orig_height': orig_h,
                'orig_width': orig_w,
                'final_height': final_image.shape[0], # Preserves original dimensions
                'final_width': final_image.shape[1], # Preserves original dimensions
                **process_meta
            }
            if fan_contour is not None:
                x, y, w, h = cv2.boundingRect(fan_contour)
                meta['fan_bbox_x'] = x
                meta['fan_bbox_y'] = y
                meta['fan_bbox_w'] = w
                meta['fan_bbox_h'] = h

            return final_image, meta
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None, None

def main_trimester2():
    parser = argparse.ArgumentParser(description="Preprocess Trimester 2 ultrasound images for segmentation.")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to the CSV file containing image metadata.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory where the raw images are stored.")
    parser.add_argument("--output_root", type=str, default="preprocessed_data_trimester2",
                        help="Root directory for saving preprocessed images and log.")
    parser.add_argument("--erosion_kernel_size", type=int, default=5,
                        help="Kernel size for erosion in fan detection. Default: 5.")
    parser.add_argument("--dilation_kernel_size", type=int, default=5,
                        help="Kernel size for dilation in fan detection. Default: 5.")
    parser.add_argument("--text_inpaint_method", type=str, default='inpaint_telea',
                        choices=['inpaint_telea', 'mask_fill'],
                        help="Method for text inpainting ('inpaint_telea' or 'mask_fill'). Default: 'inpaint_telea'.")
    parser.add_argument("--padding_color", type=int, nargs=3, default=[0, 0, 0],
                        help="RGB color for padding/masking. Default: 0 0 0 (black).")
    args = parser.parse_args()

    # Create output directories
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load image metadata
    try:
        df = pd.read_csv(args.data_csv)
        # Ensure 'abs_path' and 'video_subdir' are correctly derived or present in CSV
        # Assuming 'rel_norm' column exists which has paths like 'train/Category1/img.png'
        # or that 'video_subdir' column explicitly exists in the CSV
        if 'rel_norm' in df.columns:
            parts = df['rel_norm'].str.split('/')
            df['video_subdir'] = parts.apply(lambda x: '/'.join(x[:-1]))
            df['filename'] = parts.str[-1]
            df['abs_path'] = df['rel_norm'].apply(lambda p: os.path.join(args.data_root, p))
        elif not all(col in df.columns for col in ['abs_path', 'video_subdir', 'filename']):
             logger.error("CSV must contain 'rel_norm' or 'abs_path', 'video_subdir', 'filename' columns.")
             return
    except FileNotFoundError:
        logger.error(f"Error: data_csv file not found at {args.data_csv}")
        return
    except Exception as e:
        logger.error(f"Error loading or parsing data_csv: {e}")
        return

    pre = UltrasoundPreprocessorTrimester2(
        erosion_kernel_size=args.erosion_kernel_size,
        dilation_kernel_size=args.dilation_kernel_size,
        text_inpaint_method=args.text_inpaint_method,
        padding_color=tuple(args.padding_color)
    )

    records = []
    logger.info(f"Starting preprocessing for {len(df)} images (Trimester 2).")

    # Process each entry
    for idx, row in df.iterrows():
        src_path = Path(row['abs_path'])
        
        # Use the stored 'video_subdir' to reconstruct the full destination directory path
        dst_dir = output_root / row['video_subdir']
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            img_out, process_meta = pre.preprocess(src_path)
            if img_out is None or img_out.size == 0:
                logger.warning(f"Processing resulted in an empty image for {src_path}. Skipping save.")
                continue

            out_path = dst_dir / row['filename']
            cv2.imwrite(str(out_path), img_out)

            log = {
                **row.to_dict(), 
                **process_meta
            }
            log['output_path'] = str(out_path)
            log['parameters'] = (
                f"erosion_kernel_size={args.erosion_kernel_size}, dilation_kernel_size={args.dilation_kernel_size}, "
                f"text_inpaint_method={args.text_inpaint_method}, padding_color={args.padding_color}"
            )
            records.append(log)
        except Exception as e:
            logger.error(f"Error processing {src_path}: {e}")

    # Save the complete processing log to an Excel file
    log_df = pd.DataFrame(records)
    log_path = output_root / 'processing_log_trimester2_segmentation.xlsx' 
    log_df.to_excel(log_path, index=False)
    logger.info(f"Preprocessing complete. Log saved to {log_path}")

if __name__ == "__main__":
    main_trimester2()