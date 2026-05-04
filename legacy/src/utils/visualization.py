import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE

# Add project root to path for absolute imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import LOGS_DIR

def plot_training_curves(train_losses, val_losses, metrics=None, save_path=None):
    """
    Plots training and validation loss curves, and optionally other metrics.

    Args:
        train_losses (list[float]): List of training loss values per epoch.
        val_losses (list[float]): List of validation loss values per epoch.
        metrics (dict[str, list[float]], optional): Dictionary of additional metrics.
        save_path (str, optional): Path to save the plot as a file.
    """
    # Create figure with appropriate size and subplots
    nrows = 1 + (1 if metrics else 0)
    fig, axes = plt.subplots(nrows=nrows, figsize=(10, 4 * nrows))
    
    # If only one subplot, convert to list for consistent indexing
    if nrows == 1:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(train_losses, label="Training Loss", color="blue")
    axes[0].plot(val_losses, label="Validation Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics if provided
    if metrics and len(metrics) > 0:
        ax = axes[1]
        for metric_name, metric_values in metrics.items():
            ax.plot(metric_values, label=metric_name.capitalize())
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plots the confusion matrix for sentiment predictions.

    Args:
        y_true (list or np.ndarray): Ground truth sentiment values.
        y_pred (list or np.ndarray): Predicted sentiment values.
        labels (list[str], optional): Class labels.
        save_path (str, optional): Path to save the plot as a file.
    """
    # Convert continuous sentiment to binary or categorical if needed
    if labels is None:
        # Binary sentiment (positive/negative)
        y_true_bin = [1 if y > 0 else 0 for y in y_true]
        y_pred_bin = [1 if y > 0 else 0 for y in y_pred]
        labels = ["Negative", "Positive"]
    else:
        y_true_bin = y_true
        y_pred_bin = y_pred
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Sentiment Classification Confusion Matrix")
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_scatter_predictions(y_pred, y_true, save_path=None, title=None):
    """
    Plots a scatter plot comparing true vs. predicted sentiment values.

    Args:
        y_true (list or np.ndarray): Ground truth sentiment values.
        y_pred (list or np.ndarray): Predicted sentiment values.
        save_path (str, optional): Path to save the plot as a file.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot scatter
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    
    # Add details
    plt.xlabel("True Sentiment")
    plt.ylabel("Predicted Sentiment")
    if title:
        plt.title(title)
    else:
        plt.title("True vs Predicted Sentiment Values")
    plt.grid(True)
    
    # Add correlation coefficient
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.annotate(f"Correlation: {corr:.3f}", 
                 xy=(0.05, 0.95), 
                 xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    
    plt.show()

def plot_features_tsne(features, labels, modality=None, save_path=None):
    """
    Visualizes features using t-SNE with color based on sentiment labels.

    Args:
        features (np.ndarray): Feature matrix.
        labels (list or np.ndarray): Sentiment values.
        modality (str, optional): Name of the modality (for title).
        save_path (str, optional): Path to save the plot as a file.
    """
    # Convert labels to categories for coloring
    # Bin sentiment scores into 7 categories from -3 to +3
    sentiment_bins = np.linspace(-3, 3, 7)
    binned_labels = np.digitize(labels, sentiment_bins) - 1
    
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot with colormap based on sentiment
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=labels, 
        cmap="coolwarm", 
        alpha=0.7,
        s=50
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sentiment Score")
    
    # Set title and labels
    title = f"t-SNE Visualization of {modality.capitalize()} Features" if modality else "t-SNE Visualization of Features"
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")
    
    plt.show()

def plot_modality_contributions(contributions, save_path=None):
    """
    Plots modality contribution scores as a horizontal bar chart.

    Args:
        contributions (dict[str, float]): Dictionary mapping modality to score.
        save_path (str, optional): Path to save the plot as a file.
    """
    modalities = list(contributions.keys())
    scores = list(contributions.values())
    
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(modalities))
    plt.barh(y_pos, scores, align="center", alpha=0.8)
    plt.yticks(y_pos, [m.capitalize() for m in modalities])
    
    # Add details
    plt.xlabel("Contribution Score")
    plt.title("Modality Contributions to Sentiment Prediction")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Modality contributions plot saved to {save_path}")
    
    plt.show()

def visualize_attention_weights(attention_weights, modalities=None, save_path=None):
    """
    Visualizes cross-modal attention weights as a heatmap.

    Args:
        attention_weights (np.ndarray): Square matrix of attention weights.
        modalities (list[str], optional): List of modality names.
        save_path (str, optional): Path to save the plot as a file.
    """
    if modalities is None:
        modalities = ["Text", "Audio", "Visual"]
    
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        annot=True,
        cmap="Blues",
        xticklabels=modalities,
        yticklabels=modalities,
        fmt=".2f"
    )
    
    plt.title("Cross-Modal Attention Weights")
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Attention weights plot saved to {save_path}")
    
    plt.show()

def visualize_results_summary(metrics_dict, model_name, save_path=None):
    """
    Plots summary bar chart of evaluation metrics for a model.

    Args:
        metrics_dict (dict[str, float]): Metric names and values.
        model_name (str): Name of the model (for title).
        save_path (str, optional): Path to save the plot as a file.
    """
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    plt.bar(metrics, values, alpha=0.8)
    
    # Add metric values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    
    # Add details
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title(f"Performance Metrics for {model_name}")
    plt.ylim(0, max(values) * 1.2)  # Add some space above bars
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Results summary saved to {save_path}")
    
    plt.show()

def setup_plotting_directory(experiment_name):
    """
    Creates a directory structure for storing plots for a given experiment.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        Path: Path to the plots directory.
    """
    plot_dir = LOGS_DIR / experiment_name / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    return plot_dir