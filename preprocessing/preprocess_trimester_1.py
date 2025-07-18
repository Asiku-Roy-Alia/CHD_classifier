import argparse
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging for preprocessing script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltrasoundPreprocessorTrimester1:
    """
    Preprocesses fetal echocardiography frames for Trimester 1 (classification model).
    This includes cropping fixed overlays, inpainting residual text, extracting
    the ultrasound fan/sector, keeping the full image (no panel splitting),
    enhancing B-mode contrast via CLAHE, and resizing with aspect-ratio preservation and padding.
    """
    def __init__(self,
                 target_size=(256, 256),
                 top_frac=0.18,
                 side_frac=0.05,
                 bottom_frac=0.05,
                 text_inpaint=True,
                 clahe=True,
                 padding_color=(0, 0, 0)):
        """
        Initializes the preprocessor with various parameters for image manipulation.

        Args:
            target_size (tuple): Desired output image size (width, height).
            top_frac (float): Fraction of the image to crop from the top.
            side_frac (float): Fraction of the image to crop from each side.
            bottom_frac (float): Fraction of the image to crop from the bottom.
            text_inpaint (bool): Whether to inpaint (remove) text overlays.
            clahe (bool): Whether to apply Contrast Limited Adaptive Histogram Equalization.
            padding_color (tuple): RGB color for padding if resizing changes aspect ratio.
        """
        self.target_size = target_size
        self.top_frac = top_frac
        self.side_frac = side_frac
        self.bottom_frac = bottom_frac
        self.text_inpaint = text_inpaint
        self.clahe = clahe
        self.padding_color = padding_color

        if clahe:
            self.clahe_proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe_proc = None

    def _crop_borders(self, image):
        """Crops specified fractions from image borders."""
        h, w = image.shape[:2]
        top = int(h * self.top_frac)
        bottom = h - int(h * self.bottom_frac)
        left = int(w * self.side_frac)
        right = w - int(w * self.side_frac)
        return image[top:bottom, left:right]

    def _inpaint_text_simple(self, image):
        """Simple text inpainting using a fixed region, suitable for typical ultrasound overlays."""
        h, w = image.shape[:2]
        # Define areas likely to contain text based on common ultrasound layouts
        text_regions = [
            (0, 0, w, int(h * 0.05)),  # Top strip
            (0, int(w * 0.8), w, h),   # Right strip (time/date)
            (int(h * 0.95), 0, w, h)   # Bottom strip
        ]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for (x1, y1, x2, y2) in text_regions:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    def _extract_fan_contour(self, image):
        """
        Extracts the ultrasound fan/sector by finding the largest contour.
        Assumes the fan is the most prominent light region.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, (0, 0, image.shape[1], image.shape[0]) # Return original if no contours
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w], (x, y, w, h)

    def _apply_clahe(self, image):
        """Applies CLAHE to the grayscale version of the image."""
        if self.clahe_proc:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_gray = self.clahe_proc.apply(gray)
            return cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
        return image

    def _resize_with_padding(self, image):
        """Resizes image to target_size, maintaining aspect ratio with padding."""
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.target_size
        
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        padded_image = np.full((target_h, target_w, 3), self.padding_color, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        return padded_image

    def preprocess(self, image_path: Path):
        """
        Performs the complete preprocessing pipeline on a single image.

        Args:
            image_path (Path): Path to the input image.

        Returns:
            tuple: A tuple containing the preprocessed image (numpy array) and a dictionary
                   of metadata (original dimensions, final dimensions, fan region, parameters).
                   Returns (None, None) if preprocessing fails.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None, None

            orig_h, orig_w = image.shape[:2]
            
            # Step 1: Crop fixed overlays
            processed_image = self._crop_borders(image.copy())

            # Step 2: Inpaint residual text
            if self.text_inpaint:
                processed_image = self._inpaint_text_simple(processed_image)

            # Step 3: Extract ultrasound fan/sector (this step crops to the fan)
            processed_image, fan_bbox = self._extract_fan_contour(processed_image)

            # Step 4: Apply CLAHE for contrast enhancement
            if self.clahe:
                processed_image = self._apply_clahe(processed_image)
            
            # Step 5: Resize with aspect-ratio preservation and padding
            final_image = self._resize_with_padding(processed_image)

            meta = {
                'orig_height': orig_h,
                'orig_width': orig_w,
                'final_height': final_image.shape[0],
                'final_width': final_image.shape[1],
                'fan_x': fan_bbox[0],
                'fan_y': fan_bbox[1],
                'fan_w': fan_bbox[2],
                'fan_h': fan_bbox[3]
            }
            return final_image, meta
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None, None

def main_trimester1():
    parser = argparse.ArgumentParser(description="Preprocess Trimester 1 ultrasound images for classification.")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to the CSV file containing image metadata (e.g., 'image_data.csv').")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory where the raw images are stored.")
    parser.add_argument("--output_root", type=str, default="preprocessed_data_trimester1",
                        help="Root directory for saving preprocessed images and log.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Desired output image size (width height). Default: 256 256.")
    parser.add_argument("--top_frac", type=float, default=0.18,
                        help="Fraction of image to crop from top. Default: 0.18.")
    parser.add_argument("--side_frac", type=float, default=0.05,
                        help="Fraction of image to crop from sides. Default: 0.05.")
    parser.add_argument("--bottom_frac", type=float, default=0.05,
                        help="Fraction of image to crop from bottom. Default: 0.05.")
    parser.add_argument("--no_text_inpaint", action="store_false", dest="text_inpaint",
                        help="Disable text inpainting.")
    parser.add_argument("--no_clahe", action="store_false", dest="clahe",
                        help="Disable CLAHE contrast enhancement.")
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

    pre = UltrasoundPreprocessorTrimester1(
        target_size=tuple(args.target_size),
        top_frac=args.top_frac,
        side_frac=args.side_frac,
        bottom_frac=args.bottom_frac,
        text_inpaint=args.text_inpaint,
        clahe=args.clahe
    )

    records = []
    logger.info(f"Starting preprocessing for {len(df)} images (Trimester 1).")

    # Process each entry
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Images"):
        src = Path(row['abs_path'])
        
        # Use the stored 'video_subdir' to reconstruct the full destination directory path
        dst_dir = output_root / row['video_subdir']
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            img_out, meta = pre.preprocess(src)
            if img_out is None:
                logger.warning(f"Preprocessing failed for {src}. Skipping.")
                continue

            out_path = dst_dir / row['filename']
            cv2.imwrite(str(out_path), img_out)
            
            log = {**row.to_dict(), **meta}
            log['output_path'] = str(out_path)
            log['parameters']  = (
                f"target_size={args.target_size}, top_frac={args.top_frac}, side_frac={args.side_frac}, "
                f"bottom_frac={args.bottom_frac}, text_inpaint={args.text_inpaint}, clahe={args.clahe}"
            )
            records.append(log)
        except Exception as e:
            logger.error(f"Error during preprocessing {src}: {e}")

    # Save processing log
    log_df = pd.DataFrame(records)
    log_path = output_root / 'processing_log_trimester1_classification.xlsx'
    log_df.to_excel(log_path, index=False)
    logger.info(f"Preprocessing complete. Log saved to {log_path}")

if __name__ == "__main__":
    main_trimester1()
