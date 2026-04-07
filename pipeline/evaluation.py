"""Evaluation and comparison utilities for model assessment."""

import torch
import numpy as np
from typing import Dict, List, Tuple
import torchmetrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for flood prediction.
    
    Metrics include:
    - Regression metrics: MSE, MAE, RMSE, R²
    - Classification metrics (after thresholding): Precision, Recall, F1
    - Flood-specific metrics: Critical Success Index (CSI), Bias
    
    Args:
        y_true: True fractional inundation values (N,) or (N, H, W)
        y_pred: Predicted fractional inundation values (N,) or (N, H, W)
        threshold: Threshold for converting fractional to binary labels
    
    Returns:
        Dictionary of computed metrics
    """
    # Flatten if needed
    y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    
    # Regression metrics
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Classification metrics (after thresholding)
    y_true_binary = (y_true_flat >= threshold).astype(int)
    y_pred_binary = (y_pred_flat >= threshold).astype(int)
    
    # Confusion matrix components
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Flood-specific metrics
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # Critical Success Index
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0  # Bias score
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'csi': float(csi),
        'bias': float(bias),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple model configurations or baselines.
    
    Args:
        results: Dictionary mapping model names to their metric dictionaries
        save_path: Optional path to save comparison visualization
    """
    # Extract metrics
    metrics = ['mse', 'mae', 'rmse', 'r2', 'precision', 'recall', 'f1', 'csi']
    model_names = list(results.keys())
    
    # Create comparison DataFrame
    comparison_data = {metric: [results[model][metric] for model in model_names] 
                       for metric in metrics}
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[model][metric] for model in model_names]
        ax.bar(model_names, values)
        ax.set_title(metric.upper())
        ax.set_ylabel('Score')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def ablation_study_report(
    baseline_metrics: Dict[str, float],
    ablation_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Generate report comparing baseline model with ablation study variants.
    
    Demonstrates the contribution of different components (e.g., HAND, temporal features).
    
    Args:
        baseline_metrics: Metrics for full model
        ablation_results: Dictionary mapping ablation names to their metrics
        save_path: Optional path to save report
    """
    print("=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"\nBaseline (Full Model):")
    for metric, value in baseline_metrics.items():
        if metric not in ['tp', 'fp', 'fn', 'tn']:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nAblation Variants:")
    for variant_name, metrics in ablation_results.items():
        print(f"\n  {variant_name}:")
        for metric, value in metrics.items():
            if metric not in ['tp', 'fp', 'fn', 'tn']:
                change = value - baseline_metrics[metric]
                change_pct = (change / baseline_metrics[metric] * 100) if baseline_metrics[metric] != 0 else 0
                print(f"    {metric.upper()}: {value:.4f} ({change:+.4f}, {change_pct:+.2f}%)")


def evaluate_across_regions(
    model: torch.nn.Module,
    test_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: str = 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across different geographic regions.
    
    Demonstrates generalizability and identifies potential regional biases.
    
    Args:
        model: Trained model
        test_datasets: Dictionary mapping region names to (X_test, y_test) tuples
        device: Device to run evaluation on
    
    Returns:
        Dictionary mapping region names to their evaluation metrics
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for region_name, (X_test, y_test) in test_datasets.items():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred = model(X_tensor).cpu().numpy()
            
            metrics = compute_comprehensive_metrics(y_test, y_pred)
            results[region_name] = metrics
    
    return results
