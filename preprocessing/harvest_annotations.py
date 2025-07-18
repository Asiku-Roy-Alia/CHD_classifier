import argparse
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnotationHarvester:
    """
    Harvests annotation masks from a specified directory, processes them (e.g., ensures
    correct format), and generates a structured metadata log.
    Assumes annotation masks are image files (e.g., PNG) where pixel values
    represent class IDs.
    """
    def __init__(self, output_mask_format='PNG', output_mask_dtype=np.uint8):
        """
        Initializes the AnnotationHarvester.

        Args:
            output_mask_format (str): Desired output format for processed masks (e.g., 'PNG').
            output_mask_dtype (numpy.dtype): Desired data type for processed mask pixel values.
                                              Use np.uint8 for 0-255 class IDs.
        """
        self.output_mask_format = output_mask_format
        self.output_mask_dtype = output_mask_dtype

    def process_mask(self, mask_path: Path):
        """
        Loads an annotation mask and ensures it's in the correct format and data type.

        Args:
            mask_path (Path): Path to the input annotation mask image.

        Returns:
            numpy.ndarray: Processed mask array, or None if loading fails.
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) # Read as is, preserving channels
            if mask is None:
                logger.warning(f"Could not read mask image: {mask_path}")
                return None
            
            # Convert to grayscale if it's a multi-channel image (e.g., RGB mask)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Ensure the correct data type
            if mask.dtype != self.output_mask_dtype:
                mask = mask.astype(self.output_mask_dtype)
            
            return mask
        except Exception as e:
            logger.error(f"Error processing mask {mask_path}: {e}")
            return None

def main_harvest_annotations():
    parser = argparse.ArgumentParser(description="Harvest and process annotation masks.")
    parser.add_argument("--annotations_root", type=str, required=True,
                        help="Root directory containing raw annotation mask files.")
    parser.add_argument("--output_root", type=str, default="processed_annotations",
                        help="Root directory for saving processed annotation masks and log.")
    parser.add_argument("--image_extensions", type=str, nargs='+', default=['.png', '.jpg', '.jpeg'],
                        help="List of image file extensions to search for annotations. Default: .png .jpg .jpeg.")
    parser.add_argument("--output_mask_format", type=str, default='png',
                        help="Output file format for processed masks (e.g., 'png'). Default: png.")
    args = parser.parse_args()

    # Create output directories
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    harvester = AnnotationHarvester(output_mask_format=args.output_mask_format)

    records = []
    logger.info(f"Starting annotation harvesting from {args.annotations_root}.")

    # Recursively find all annotation files
    annotation_files = []
    for ext in args.image_extensions:
        annotation_files.extend(list(Path(args.annotations_root).rglob(f"*{ext}")))
    
    if not annotation_files:
        logger.warning(f"No annotation files found with extensions {args.image_extensions} in {args.annotations_root}.")
        return

    for annotation_path in tqdm(annotation_files, desc="Harvesting Annotations"):
        try:
            # Determine relative path to maintain directory structure in output
            relative_path = annotation_path.relative_to(args.annotations_root)
            output_dir = output_root / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Construct output filename with desired format
            output_filename = annotation_path.stem + '.' + args.output_mask_format
            output_path = output_dir / output_filename

            processed_mask = harvester.process_mask(annotation_path)
            if processed_mask is None:
                continue # Skip if processing failed

            cv2.imwrite(str(output_path), processed_mask)

            log = {
                'original_path': str(annotation_path),
                'processed_path': str(output_path),
                'original_height': processed_mask.shape[0],
                'original_width': processed_mask.shape[1],
                'output_format': args.output_mask_format
            }
            records.append(log)

        except Exception as e:
            logger.error(f"Error processing {annotation_path}: {e}")

    # Save processing log
    log_df = pd.DataFrame(records)
    log_path = output_root / 'annotation_harvesting_log.xlsx'
    log_df.to_excel(log_path, index=False)
    logger.info(f"Annotation harvesting complete. Log saved to {log_path}")

if __name__ == "__main__":
    main_harvest_annotations()
