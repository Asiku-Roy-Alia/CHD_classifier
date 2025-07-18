import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging
from typing import List, Dict, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_combine_fold_data(base_path: Path, num_folds: int) -> pd.DataFrame:
    """
    Loads training history and evaluation results from multiple cross-validation folds
    and combines them into a single pandas DataFrame.

    Args:
        base_path (Path): The base directory where fold results are stored (e.g., 'classification_checkpoints').
                          Each fold should have a subdirectory like 'fold_1', 'fold_2', etc.,
                          containing 'training_history.json' and 'evaluation_results.json'.
        num_folds (int): The number of cross-validation folds.

    Returns:
        pd.DataFrame: A combined DataFrame containing metrics for all epochs across all folds.
                      Returns an empty DataFrame if no data is found.
    """
    all_data = []
    for fold_idx in range(1, num_folds + 1):
        fold_dir = base_path / f'fold_{fold_idx}'
        history_path = fold_dir / 'training_history.json' # Assuming history is saved as JSON
        eval_path = fold_dir / 'evaluation_results.json'

        fold_metrics = {}
        
        # Load training history
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                # Training history might contain lists of metrics per epoch
                epochs = len(history.get('train_losses', []))
                for epoch in range(epochs):
                    epoch_data = {
                        'Fold': fold_idx,
                        'Epoch': epoch + 1,
                        'Train Loss': history['train_losses'][epoch],
                        'Val Loss': history['val_losses'][epoch],
                        'Val Accuracy': history['val_accuracies'][epoch],
                        'Val F1 Weighted': history['val_f1_scores'][epoch]
                    }
                    all_data.append(epoch_data)
            except Exception as e:
                logger.warning(f"Could not load training history for fold {fold_idx} from {history_path}: {e}")
        else:
            logger.warning(f"Training history not found for fold {fold_idx} at {history_path}")

        # Load evaluation results (final metrics for the fold)
        if eval_path.exists():
            try:
                with open(eval_path, 'r') as f:
                    eval_results = json.load(f)
                # Add final evaluation metrics to the last epoch of the fold
                if all_data:
                    last_epoch_data = all_data[-1]
                    if last_epoch_data['Fold'] == fold_idx: # Ensure it's the correct fold's last epoch
                        last_epoch_data['Final Accuracy'] = eval_results.get('accuracy')
                        last_epoch_data['Final F1 Weighted'] = eval_results.get('f1_weighted')
                        last_epoch_data['Final F1 Macro'] = eval_results.get('f1_macro')
                        # Add AUC scores if present
                        for class_name, auc_score in eval_results.get('auc_scores', {}).items():
                            last_epoch_data[f'AUC {class_name}'] = auc_score
            except Exception as e:
                logger.warning(f"Could not load evaluation results for fold {fold_idx} from {eval_path}: {e}")
        else:
            logger.warning(f"Evaluation results not found for fold {fold_idx} at {eval_path}")

    if not all_data:
        logger.warning("No data found across all folds. Ensure JSON history and evaluation files exist.")
        return pd.DataFrame()

    combined_df = pd.DataFrame(all_data)
    logger.info(f"Loaded combined data with shape: {combined_df.shape}")
    logger.info(f"Available columns: {combined_df.columns.tolist()}")
    return combined_df

def create_publication_plots(df: pd.DataFrame, save_dir: Path):
    """
    Generates various publication-ready plots.

    Args:
        df (pd.DataFrame): Combined DataFrame containing all fold data.
        save_dir (Path): Directory to save the generated plots.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating plots and saving to: {save_dir}")

    if df.empty:
        logger.warning("DataFrame is empty, skipping plot generation.")
        return

    # Ensure 'Epoch' column exists for plotting
    if 'Epoch' not in df.columns:
        logger.error("Missing 'Epoch' column in DataFrame. Cannot generate epoch-based plots.")
        return

    # --- 1. Convergence Analysis (Training/Validation Loss and Metrics over Epochs) ---
    # Plotting mean and standard deviation across folds
    metrics_to_plot = {
        'Loss': ['Train Loss', 'Val Loss'],
        'Accuracy': ['Val Accuracy'],
        'F1 Score': ['Val F1 Weighted']
    }

    for metric_type, metric_cols in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        for col in metric_cols:
            if col in df.columns:
                mean_metric = df.groupby('Epoch')[col].mean()
                std_metric = df.groupby('Epoch')[col].std()
                plt.plot(mean_metric.index, mean_metric, label=f'Mean {col}')
                plt.fill_between(mean_metric.index, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2, label=f'Std Dev {col}')
        plt.title(f'Convergence Analysis: Mean {metric_type} over Epochs (with Std Dev)')
        plt.xlabel('Epoch')
        plt.ylabel(metric_type)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'convergence_analysis_{metric_type.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
        logger.info(f"Generated convergence plot for {metric_type}.")

    # --- 2. Distribution Analysis (Box Plots/Violin Plots for Final Metrics) ---
    final_metrics = [col for col in df.columns if col.startswith('Final ') or col.startswith('AUC ')]
    if not final_metrics:
        logger.warning("No 'Final' or 'AUC' metrics found for distribution analysis.")
    else:
        # Filter to only the last epoch of each fold for final metrics
        final_epoch_df = df.drop_duplicates(subset=['Fold'], keep='last')

        if not final_epoch_df.empty:
            plt.figure(figsize=(12, len(final_metrics) * 1.5))
            sns.boxplot(data=final_epoch_df[final_metrics], orient='h', palette='viridis')
            plt.title('Distribution of Final Performance Metrics Across Folds')
            plt.xlabel('Metric Value')
            plt.tight_layout()
            plt.savefig(save_dir / 'distribution_analysis_boxplot.png', dpi=300)
            plt.close()
            logger.info("Generated distribution analysis box plot.")

            plt.figure(figsize=(12, len(final_metrics) * 1.5))
            sns.violinplot(data=final_epoch_df[final_metrics], orient='h', palette='plasma')
            plt.title('Distribution of Final Performance Metrics Across Folds (Violin Plot)')
            plt.xlabel('Metric Value')
            plt.tight_layout()
            plt.savefig(save_dir / 'distribution_analysis_violinplot.png', dpi=300)
            plt.close()
            logger.info("Generated distribution analysis violin plot.")
        else:
            logger.warning("Final epoch data is empty for distribution analysis.")

    # --- 3. Correlation Analysis (Heatmap of Inter-Metric Correlations) ---
    # Consider only numerical columns for correlation
    numerical_df = df.select_dtypes(include=np.number)
    if not numerical_df.empty and numerical_df.shape[1] > 1:
        corr_matrix = numerical_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Performance Metrics')
        plt.tight_layout()
        plt.savefig(save_dir / 'correlation_analysis_heatmap.png', dpi=300)
        plt.close()
        logger.info("Generated correlation analysis heatmap.")
    else:
        logger.warning("Not enough numerical columns for correlation analysis.")

    # --- 4. Stability Analysis (Coefficient of Variation and Final Performance Comparison) ---
    if not final_epoch_df.empty:
        stability_df = pd.DataFrame(index=final_metrics)
        stability_df['Mean'] = final_epoch_df[final_metrics].mean()
        stability_df['Std Dev'] = final_epoch_df[final_metrics].std()
        stability_df['Coefficient of Variation (%)'] = (stability_df['Std Dev'] / stability_df['Mean']) * 100
        
        logger.info("\nStability Analysis - Mean, Std Dev, and Coefficient of Variation:\n")
        logger.info(stability_df.to_string())

        # Plotting final performance comparison (e.g., bar plot of means with error bars)
        plt.figure(figsize=(12, 7))
        stability_df['Mean'].plot(kind='bar', yerr=stability_df['Std Dev'], capsize=5, color='skyblue')
        plt.title('Mean Final Performance Across Folds (with Std Dev)')
        plt.ylabel('Metric Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'stability_analysis_final_performance.png', dpi=300)
        plt.close()
        logger.info("Generated stability analysis plot.")
    else:
        logger.warning("No final epoch data for stability analysis.")

    # --- 5. Learning Curves (Trend Fitting) ---
    # This is partially covered by convergence analysis. For explicit trend fitting,
    # you might fit a curve (e.g., polynomial) to the mean training/validation loss.
    # For simplicity, relying on the convergence plots for now.
    # If more complex learning curve analysis is needed (e.g., varying dataset size),
    # the data input would need to reflect that.

    # --- 6. Radar Chart (Comprehensive Multi-dimensional Performance Comparison) ---
    if not final_epoch_df.empty and len(final_metrics) >= 3: # Need at least 3 metrics for a radar chart
        # Select a subset of key final metrics for the radar chart
        radar_metrics = ['Final Accuracy', 'Final F1 Weighted']
        # Add AUCs if available and relevant
        for col in final_metrics:
            if col.startswith('AUC'):
                radar_metrics.append(col)
        
        # Ensure selected metrics are actually in the DataFrame
        radar_metrics = [m for m in radar_metrics if m in final_epoch_df.columns]
        
        if len(radar_metrics) >= 3:
            # Calculate mean values for the radar chart
            mean_values = final_epoch_df[radar_metrics].mean().values
            
            # Normalize values to 0-1 range for radar chart, or use appropriate max values
            # For metrics like Accuracy, F1, AUC, max is 1.0. If other metrics are included, normalize.
            max_values = np.ones_like(mean_values) # Assuming max is 1.0 for these metrics
            
            # Number of variables
            num_vars = len(radar_metrics)
            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            
            # The plot must be circular, so add the first value to the end
            values = mean_values.tolist()
            values += values[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=2, linestyle='solid', label='Mean Performance')
            ax.fill(angles, values, 'blue', alpha=0.25)
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines and labels
            ax.set_rlabel_position(0)
            # Set ticks to 0.2, 0.4, 0.6, 0.8, 1.0 (or adjust based on your metric range)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=8)
            ax.set_ylim(0, 1) # Set limits from 0 to 1

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_metrics)

            plt.title('Radar Chart: Mean Performance Across Key Metrics', size=16, color='blue', y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.savefig(save_dir / 'radar_chart_performance.png', dpi=300)
            plt.close()
            logger.info("Generated radar chart.")
        else:
            logger.warning("Not enough final metrics (at least 3) to generate radar chart.")

def generate_statistical_summary(df: pd.DataFrame, save_dir: Path):
    """
    Generates a detailed statistical summary of the performance metrics.

    Args:
        df (pd.DataFrame): Combined DataFrame containing all fold data.
        save_dir (Path): Directory to save the summary text file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / 'statistical_summary.txt'

    if df.empty:
        logger.warning("DataFrame is empty, skipping statistical summary generation.")
        with open(summary_path, 'w') as f:
            f.write("No data available for statistical summary.\n")
        return

    # Filter to only the last epoch of each fold for final metrics
    final_epoch_df = df.drop_duplicates(subset=['Fold'], keep='last')

    if final_epoch_df.empty:
        logger.warning("No final epoch data found for statistical summary.")
        with open(summary_path, 'w') as f:
            f.write("No final epoch data available for statistical summary.\n")
        return

    with open(summary_path, 'w') as f:
        f.write("--- Statistical Summary of Model Performance Across Folds ---\n\n")
        f.write(f"Number of Folds: {df['Fold'].nunique()}\n")
        f.write(f"Total Epochs per Fold (if available): {df['Epoch'].max() if 'Epoch' in df.columns else 'N/A'}\n\n")

        f.write("Overall Performance Metrics (Mean ± Std Dev across Folds):\n")
        
        # Select relevant final metrics for summary
        metrics_for_summary = [col for col in final_epoch_df.columns if col.startswith('Final ') or col.startswith('AUC ')]
        
        if not metrics_for_summary:
            f.write("No 'Final' or 'AUC' metrics found for detailed summary.\n")
        else:
            for metric in metrics_for_summary:
                mean_val = final_epoch_df[metric].mean()
                std_val = final_epoch_df[metric].std()
                f.write(f"- {metric}: {mean_val:.4f} ± {std_val:.4f}\n")
            f.write("\n")

            f.write("Detailed Statistics per Metric (across Folds):\n")
            f.write(final_epoch_df[metrics_for_summary].describe().to_string())
            f.write("\n\n")

            f.write("Coefficient of Variation (%):\n")
            cov_df = (final_epoch_df[metrics_for_summary].std() / final_epoch_df[metrics_for_summary].mean()) * 100
            f.write(cov_df.to_string())
            f.write("\n")
        
        f.write("\n--- End of Summary ---\n")
    
    logger.info(f"Statistical summary saved to: {summary_path}")

def main_performance_graphs():
    parser = argparse.ArgumentParser(description="Generate performance graphs and statistical summaries.")
    parser.add_argument("--data_base_dir", type=str, required=True,
                        help="Base directory containing fold-wise results (e.g., 'classification_checkpoints' or 'segmentation_checkpoints').")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="Number of cross-validation folds to load data from. Default: 5.")
    parser.add_argument("--output_dir", type=str, default="analysis_plots",
                        help="Directory to save the generated plots and summary files. Default: analysis_plots.")
    args = parser.parse_args()

    data_base_dir = Path(args.data_base_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Loading and processing data from {data_base_dir} for {args.num_folds} folds...")
    combined_data = load_and_combine_fold_data(data_base_dir, args.num_folds)

    if combined_data.empty:
        logger.error("Failed to load any data. Please check data_base_dir and num_folds.")
        return

    logger.info("\nGenerating visualizations...")
    create_publication_plots(combined_data, output_dir)
    
    logger.info("\nGenerating statistical summary...")
    generate_statistical_summary(combined_data, output_dir)
    
    logger.info(f"\nAnalysis complete! All plots and summary saved in '{output_dir}' directory.")
    logger.info("Generated visualizations include:")
    logger.info("1. Convergence Analysis - Multi-fold training curves with confidence intervals")
    logger.info("2. Distribution Analysis - Box plots and violin plots for statistical distribution")
    logger.info("3. Correlation Analysis - Heatmaps showing inter-metric correlations")
    logger.info("4. Stability Analysis - Coefficient of variation and final performance comparison")
    logger.info("5. Learning Curves (covered by Convergence Analysis in this script)")
    logger.info("6. Radar Chart - Comprehensive multi-dimensional performance comparison")

if __name__ == "__main__":
    main_performance_graphs()

