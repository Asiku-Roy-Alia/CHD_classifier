import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
import logging

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_tree(base_path, prefix=""):
    """
    Recursively generates an ASCII tree structure for a directory.

    Args:
        base_path (Path): The root path to start generating the tree from.
        prefix (str): Prefix for current line to maintain tree structure.

    Returns:
        list: A list of strings, each representing one line of the tree.
    """
    lines = []
    try:
        # Get sorted entries (directories first, then files)
        entries = sorted(os.listdir(base_path), key=lambda x: (not Path(base_path, x).is_dir(), x.lower()))
    except FileNotFoundError:
        return [f"{prefix}✖ Directory not found: {base_path}"]
    except Exception as e:
        logger.error(f"Error accessing directory {base_path}: {e}")
        return [f"{prefix}✖ Error: {e}"]

    for index, entry in enumerate(entries):
        path = base_path / entry
        is_last = (index == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry}")
        if path.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(generate_tree(path, prefix + extension))
    return lines

def analyze_classification_counts(tree_lines, splits, classes, image_extensions):
    """
    Analyzes a dataset tree (list of lines) to count images per class and split
    for a classification-style dataset structure (e.g., split/class/image.jpg).

    Args:
        tree_lines (list): List of strings representing the dataset tree.
        splits (list): List of expected split names (e.g., ['train', 'valid', 'test']).
        classes (list): List of expected class names.
        image_extensions (list): List of image file extensions (e.g., ['.jpg', '.png']).

    Returns:
        pandas.DataFrame: DataFrame with counts per class per split, and total.
    """
    counts = defaultdict(Counter)
    current_split = None
    current_class = None

    for line in tree_lines:
        stripped = line.strip()
        
        # Detect split folders (e.g., '├── train/')
        for split_name in splits:
            if f"├── {split_name}" in stripped or f"└── {split_name}" in stripped:
                # Ensure it's exactly the split folder, not part of a filename
                if len(stripped.split('──')[-1].strip()) == len(split_name):
                    current_split = split_name
                break
        else: # No break, meaning no split was found in this line
            # Determine tree level by indentation to find class directories and image files
            level_pos = -1
            if '├──' in line:
                level_pos = line.find('├──')
            elif '└──' in line:
                level_pos = line.find('└──')
            
            if level_pos != -1:
                item_name = stripped.split('──')[-1].strip()
                
                # Level 1 (e.g., '│   ├── ClassA/'): class directories within a split
                # Assuming 4 spaces for the class level relative to the split
                if level_pos == 4 and item_name in classes:
                    current_class = item_name
                # Level 2 (e.g., '│   │   ├── image1.jpg'): image files within a class
                # Assuming 8 spaces for the image file level relative to the split
                elif level_pos == 8 and any(item_name.lower().endswith(ext) for ext in image_extensions):
                    if current_split and current_class:
                        counts[current_split][current_class] += 1
    
    # Build DataFrame, reindex to ensure all classes are present, fill missing with 0
    counts_df = pd.DataFrame(counts).reindex(classes).fillna(0).astype(int)
    counts_df['total'] = counts_df.sum(axis=1)
    return counts_df

def main_explore_trimester1_structure():
    parser = argparse.ArgumentParser(description="Explore Trimester 1 dataset structure, generate tree, and audit counts for classification.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the Trimester 1 dataset to explore (e.g., 'data/Trimester_1_Classification').")
    parser.add_argument("--output_dir", type=str, default="documentation/from_code",
                        help="Directory to save the generated tree file and plots.")
    parser.add_argument("--splits", type=str, nargs='*', default=['train', 'valid', 'test'],
                        help="List of expected dataset splits (e.g., train valid test).")
    parser.add_argument("--classes", type=str, nargs='*', default=['Aorta', 'Flows', 'Other', 'V sign', 'X sign'],
                        help="List of expected class names in the dataset.")
    parser.add_argument("--image_extensions", type=str, nargs='*', default=['.jpg', '.jpeg', '.png'],
                        help="List of image file extensions to count. Default: .jpg .jpeg .png.")
    parser.add_argument("--generate_tree_file", action="store_true",
                        help="Generate and save the full ASCII tree to a text file.")
    parser.add_argument("--audit_counts", action="store_true",
                        help="Perform and display image counts per class and split, and generate plots.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exploring Trimester 1 dataset structure for: {data_root}")

    # 1. Generate and optionally save the directory tree
    tree_lines = [data_root.name] + generate_tree(data_root)
    
    if args.generate_tree_file:
        tree_output_path = output_dir / f"{data_root.name.replace(os.sep, '_')}_tree.txt"
        with open(tree_output_path, "w", encoding="utf-8") as f:
            for line in tree_lines:
                f.write(line + "\n")
        logger.info(f"Full dataset tree saved to: {tree_output_path}")
    
    print(f"\n=== Trimester 1 Dataset Tree Preview (first 15 lines) ===")
    for line in tree_lines[:15]:
        print(line)
    if len(tree_lines) > 15:
        print("...")

    # 2. Audit counts per class per split and generate plots
    if args.audit_counts:
        logger.info("Performing Trimester 1 dataset audit (counts and stratification).")
        counts_df = analyze_classification_counts(tree_lines, args.splits, args.classes, args.image_extensions)
        
        if counts_df.empty:
            logger.warning("No data found for auditing. Check --data_root, --splits, and --classes arguments.")
        else:
            print("\nCounts per class per split (Trimester 1):\n")
            print(counts_df)

            # 3. Bar plot: total images per class
            plt.figure(figsize=(10, 6))
            counts_df['total'].plot(kind='bar')
            plt.title('Trimester 1: Total Images per Class')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(output_dir / 'trimester1_total_images_per_class.png')
            plt.show()
            logger.info(f"Trimester 1 total images per class plot saved to: {output_dir / 'trimester1_total_images_per_class.png'}")

            # 4. Stratification proportions per class
            if not counts_df.empty and 'total' in counts_df.columns and (counts_df['total'] > 0).any():
                # Filter out columns that are not splits or 'total'
                split_cols = [s for s in args.splits if s in counts_df.columns]
                if split_cols:
                    props = counts_df[split_cols].div(counts_df['total'], axis=0)
                    plt.figure(figsize=(12, 7))
                    props.plot(kind='bar')
                    plt.title('Trimester 1: Train/Valid/Test Distribution per Class')
                    plt.xlabel('Class')
                    plt.ylabel('Proportion')
                    plt.legend(title='Split')
                    plt.tight_layout()
                    plt.savefig(output_dir / 'trimester1_split_distribution_per_class.png')
                    plt.show()
                    logger.info(f"Trimester 1 split distribution plot saved to: {output_dir / 'trimester1_split_distribution_per_class.png'}")
                else:
                    logger.warning("No valid split columns found for Trimester 1 stratification plot.")
            else:
                logger.warning("Trimester 1 Counts DataFrame is empty or 'total' column is missing/zero, skipping stratification plot.")

if __name__ == "__main__":
    main_explore_trimester1_structure()

