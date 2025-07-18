import argparse
import os
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

def main_explore_trimester2_structure():
    parser = argparse.ArgumentParser(description="Explore Trimester 2 dataset structure, generate tree for video frames.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the Trimester 2 dataset to explore (e.g., 'data/Trimester_2').")
    parser.add_argument("--output_dir", type=str, default="documentation/from_code",
                        help="Directory to save the generated tree file.")
    parser.add_argument("--generate_tree_file", action="store_true",
                        help="Generate and save the full ASCII tree to a text file.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exploring Trimester 2 dataset structure for: {data_root}")

    # Generate and optionally save the directory tree
    tree_lines = [data_root.name] + generate_tree(data_root)
    
    if args.generate_tree_file:
        tree_output_path = output_dir / f"{data_root.name.replace(os.sep, '_')}_tree.txt"
        with open(tree_output_path, "w", encoding="utf-8") as f:
            for line in tree_lines:
                f.write(line + "\n")
        logger.info(f"Full dataset tree saved to: {tree_output_path}")
    
    print(f"\n=== Trimester 2 Dataset Tree Preview (first 15 lines) ===")
    for line in tree_lines[:15]:
        print(line)
    if len(tree_lines) > 15:
        print("...")

    logger.info("Trimester 2 structure exploration complete. Note: Class/split counts are not performed for this video-based structure.")

if __name__ == "__main__":
    main_explore_trimester2_structure()

