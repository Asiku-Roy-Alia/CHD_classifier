import argparse
import os
import pandas as pd
from PIL import Image
import datetime
from pathlib import Path
import logging

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of image extensions to consider, including DICOM
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}

def extract_metadata(base_path: Path, dataset_name: str):
    """
    Extracts metadata for all image and DICOM files within a given base path.

    Args:
        base_path (Path): The root directory to start scanning for files.
        dataset_name (str): A name to identify this dataset in the metadata.

    Returns:
        list: A list of dictionaries, where each dictionary contains metadata
              for one image/DICOM file.
    """
    metadata = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            file_path = Path(root) / fname
            ext = file_path.suffix.lower()

            if ext in IMAGE_EXTS:
                # File system dates
                mtime = os.path.getmtime(file_path)
                file_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                width, height = None, None
                
                try:
                    if ext == ".dcm":
                        # Attempt DICOM reading (if pydicom is available)
                        import pydicom
                        ds = pydicom.dcmread(file_path)
                        if hasattr(ds, 'pixel_array'):
                            # DICOM images can be 2D or 3D (e.g., multiple frames)
                            if ds.pixel_array.ndim == 2:
                                height, width = ds.pixel_array.shape
                            elif ds.pixel_array.ndim == 3: # Handle multi-frame DICOM
                                height, width = ds.pixel_array.shape[1:] # Assuming (frames, H, W)
                        else:
                            logger.warning(f"DICOM file {file_path} has no pixel_array.")
                    else:
                        # Standard image reading
                        with Image.open(file_path) as img:
                            width, height = img.size
                except ImportError:
                    logger.warning("pydicom not installed. Cannot read DICOM files. Install with 'pip install pydicom'.")
                except Exception as e:
                    logger.warning(f"Could not read dimensions for {file_path}: {e}")

                metadata.append({
                    "dataset": dataset_name,
                    "relative_path": str(file_path.relative_to(base_path)), # Relative to the provided base_path
                    "filename": fname,
                    "extension": ext,
                    "width": width,
                    "height": height,
                    "last_modified": file_date
                })
    return metadata

def main_extract_trimester2_metadata():
    parser = argparse.ArgumentParser(description="Extract metadata from Trimester 2 image and DICOM files.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the Trimester 2 dataset from which to extract metadata (e.g., 'data/Trimester_2').")
    parser.add_argument("--dataset_name", type=str, default="Trimester_2_Video_Dataset",
                        help="A name for this dataset to be included in the metadata log.")
    parser.add_argument("--output_dir", type=str, default="documentation/from_code",
                        help="Directory to save the generated metadata Excel file.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting metadata extraction for Trimester 2 dataset '{args.dataset_name}' from: {data_root}")

    # Extract metadata
    all_metadata = extract_metadata(data_root, args.dataset_name)
    
    if not all_metadata:
        logger.warning(f"No image or DICOM files found in {data_root}.")
        return

    # Combine into a single DataFrame
    metadata_df = pd.DataFrame(all_metadata) 

    # Save the complete metadata log to an Excel file
    output_path = output_dir / f"{args.dataset_name.replace(' ', '_').lower()}_metadata.xlsx"
    metadata_df.to_excel(output_path, index=False)
    logger.info(f"Metadata extraction complete. Log saved to {output_path}")

if __name__ == "__main__":
    main_extract_trimester2_metadata()
