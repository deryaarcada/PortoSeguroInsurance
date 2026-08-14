"""
PHASE 3C: Business Impact Simulation
================================================
Loads calibrated test predictions and computes expected profit/savings 
for a grid of saved_loss and action_cost values. Saves results to CSV 
and prints a concise table for quick inspection.

Assumptions:
- For each contacted customer we pay action_cost.
- If the customer would have had a claim, we save saved_loss when we act.
- Expected savings for a customer = saved_loss * P(claim) - action_cost
- We rank customers by predicted probability and consider top-k percentiles.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any


def load_predictions(pred_path: str = 'model_results/test_preds_calibrated.npy') -> np.ndarray:
    """
    Load calibrated predictions from numpy file.
    
    Args:
        pred_path: Path to the predictions file
        
    Returns:
        Array of calibrated probabilities
        
    Raises:
        FileNotFoundError: If predictions file doesn't exist
    """
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{pred_path} not found. Run Phase 3a first.")
    return np.load(pred_path)


def load_test_ids(test_csv_path: str = 'test.csv') -> np.ndarray:
    """
    Load test IDs from CSV file.
    
    Args:
        test_csv_path: Path to the test CSV file
        
    Returns:
        Array of test IDs
    """
    if os.path.exists(test_csv_path):
        return pd.read_csv(test_csv_path)['id'].values
    return np.arange(len(probs))  # Fallback if test.csv doesn't exist


def calculate_business_impact(
    probs: np.ndarray,
    saved_losses: List[float],
    action_costs: List[float],
    k_percent_list: List[float]
) -> pd.DataFrame:
    """
    Calculate expected savings and ROI for different business scenarios.
    
    Args:
        probs: Array of predicted probabilities
        saved_losses: List of potential savings if claim is prevented
        action_costs: List of costs for taking action
        k_percent_list: List of top percentiles to target
        
    Returns:
        DataFrame with business impact metrics
    """
    results = []
    N = len(probs)
    
    # Precompute sorted indices (descending order)
    order = np.argsort(probs)[::-1]
    
    for saved in saved_losses:
        for cost in action_costs:
            for k in k_percent_list:
                # Calculate number of customers to contact
                top_n = max(1, int(N * k))
                idxs = order[:top_n]
                
                # Expected savings per contacted customer
                exp_savings_each = saved * probs[idxs] - cost
                
                # Aggregate metrics
                total_expected = exp_savings_each.sum()
                avg_per_contact = exp_savings_each.mean()
                total_cost = cost * top_n
                roi = total_expected / total_cost if total_cost > 0 else np.nan
                
                results.append({
                    'saved_loss': saved,
                    'action_cost': cost,
                    'k_percent': k,
                    'n_contacted': top_n,
                    'total_expected_saving': total_expected,
                    'avg_expected_per_contact': avg_per_contact,
                    'ROI': roi
                })
    
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_path: str = 'model_results/business_impact_sensitivity.csv') -> None:
    """
    Save results to CSV and print summary.
    
    Args:
        df: DataFrame with business impact results
        output_path: Path to save CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved sensitivity results to: {output_path}")
    
    # Print sample results as pivot table
    print('\nSample results (Total Expected Saving):')
    pivot_table = df.pivot_table(
        index=['saved_loss', 'action_cost'], 
        columns='k_percent', 
        values='total_expected_saving'
    )
    print(pivot_table)


def create_heatmaps(df: pd.DataFrame, k_percent_list: List[float]) -> None:
    """
    Create heatmaps for total expected saving at each percentile.
    
    Args:
        df: DataFrame with business impact results
        k_percent_list: List of top percentiles to visualize
    """
    heatmap_dir = 'model_results/business_impact_heatmaps'
    os.makedirs(heatmap_dir, exist_ok=True)
    
    for k in k_percent_list:
        # Pivot data for heatmap
        pivot = df[df['k_percent'] == k].pivot(
            index='saved_loss', 
            columns='action_cost', 
            values='total_expected_saving'
        )
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.0f', 
            cmap='YlGnBu', 
            cbar_kws={'label': 'Total expected saving'}
        )
        plt.title(f"Total Expected Saving (Top {k*100:.1f}% Targeted)", 
                  fontsize=12, fontweight='bold')
        plt.xlabel('Action Cost')
        plt.ylabel('Saved Loss')
        plt.tight_layout()
        
        # Save figure
        fig_path = f'{heatmap_dir}/heatmap_total_saving_{int(k*1000)}.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved heatmap: {fig_path}")


def main() -> None:
    """Main execution function."""
    # Configuration
    PRED_PATH = 'model_results/test_preds_calibrated.npy'
    TEST_CSV_PATH = 'test.csv'
    OUTPUT_PATH = 'model_results/business_impact_sensitivity.csv'
    
    # Define parameter grids
    SAVED_LOSSES = [500, 1000, 2500]
    ACTION_COSTS = [25, 50, 100]
    K_PERCENT_LIST = [0.005, 0.01, 0.02]
    
    try:
        # Load data
        print("Loading predictions...")
        probs = load_predictions(PRED_PATH)
        
        # Load test IDs (optional)
        test_ids = load_test_ids(TEST_CSV_PATH)
        print(f"Loaded {len(probs)} predictions and {len(test_ids)} test IDs")
        
        # Calculate business impact
        print("\nCalculating business impact metrics...")
        results_df = calculate_business_impact(
            probs, SAVED_LOSSES, ACTION_COSTS, K_PERCENT_LIST
        )
        
        # Save results
        save_results(results_df, OUTPUT_PATH)
        
        # Create heatmaps
        print("\nGenerating heatmaps...")
        create_heatmaps(results_df, K_PERCENT_LIST)
        
        print("\nBusiness impact sensitivity analysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
