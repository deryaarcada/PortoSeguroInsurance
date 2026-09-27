"""
PHASE 3C: Business Impact Simulation
================================================
Loads calibrated test predictions and computes expected profit/savings 
for a grid of saved_loss and action_cost values. Saves results to CSV 
and prints a concise table for quick inspection.

WHAT THIS IS — AND ISN'T
-------------------------
This is a SENSITIVITY SIMULATION under a set of explicit, stated
assumptions. It is NOT a validated business forecast, and the dollar
figures it prints should not be read as "this is how much money the
company will save." Two of the assumptions below (`effectiveness` and
`uptake_rate`) are illustrative placeholders, not numbers estimated from
this dataset or from any real intervention — they exist so the simulation
doesn't implicitly assume something unrealistic (that every contacted
customer accepts help and every claim is fully preventable). Before using
this for an actual budget decision, replace them with figures from a real
pilot program, actuarial loss-prevention data, or a vendor's track record.

Assumptions:
- For each contacted customer we pay `action_cost`.
- If the customer would have had a claim, the maximum possible saving is
  `saved_loss`. This is scaled down by two further factors before it
  counts as a realized saving:
    - `effectiveness`: even when contacted, only a fraction of claims are
      actually preventable by whatever action is taken — a phone call or
      a policy change does not stop every accident. Default 0.25 here is
      an illustrative placeholder, not a fitted value.
    - `uptake_rate`: not every contacted customer accepts the offered
      action (declines the call, ignores the recommendation, etc.).
      Default 0.70 here is likewise an illustrative placeholder.
  So: expected realized saving per customer = saved_loss * P(claim) *
  effectiveness * uptake_rate, and expected net saving = that minus
  action_cost.
- We rank customers by predicted probability and consider top-k percentiles.
- For comparison, the CSV also reports a "naive" scenario column that
  skips the effectiveness/uptake discount entirely (effectiveness=1,
  uptake_rate=1) — i.e. the "every intervention works, everyone accepts
  it" scenario from the original version of this script. Seeing both side
  by side makes clear how much of the headline saving number depends on
  those two assumptions.

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
    k_percent_list: List[float],
    effectiveness: float = 0.25,
    uptake_rate: float = 0.70,
) -> pd.DataFrame:
    """
    Calculate expected savings and ROI for different business scenarios,
    under explicit assumptions about intervention effectiveness and
    customer uptake (see module docstring for why these exist).

    Args:
        probs: Array of predicted probabilities
        saved_losses: List of potential savings if a claim is fully prevented
        action_costs: List of costs for taking action
        k_percent_list: List of top percentiles to target
        effectiveness: Fraction of claims that are actually preventable by
            the intervention, even when the customer is contacted and
            genuinely at risk (0-1). ILLUSTRATIVE PLACEHOLDER — calibrate
            against a real pilot or actuarial data before using this for
            budget decisions.
        uptake_rate: Fraction of contacted customers who accept the
            offered action (0-1). ILLUSTRATIVE PLACEHOLDER — same caveat.

    Returns:
        DataFrame with business impact metrics, including both the
        realistic (effectiveness- and uptake-adjusted) scenario and a
        "naive" scenario (effectiveness=1, uptake_rate=1) for comparison.
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

                # Maximum possible saving if the claim were fully
                # preventable and every contacted customer cooperated —
                # this is the "naive" scenario's building block.
                expected_claim_saving = saved * probs[idxs]

                # Realistic scenario: discount by effectiveness (can the
                # intervention actually prevent the claim?) and uptake
                # (does the customer go along with it?) before subtracting
                # the cost of contacting them.
                expected_prevented = expected_claim_saving * effectiveness
                expected_realized = expected_prevented * uptake_rate
                exp_savings_each = expected_realized - cost

                # Naive scenario (effectiveness=1, uptake_rate=1): kept
                # alongside the realistic one purely so the size of the
                # assumption-driven discount is visible, not to be reported
                # as a credible estimate on its own.
                naive_savings_each = expected_claim_saving - cost

                # Aggregate metrics — realistic scenario
                total_expected = exp_savings_each.sum()
                avg_per_contact = exp_savings_each.mean()
                total_cost = cost * top_n
                roi = total_expected / total_cost if total_cost > 0 else np.nan

                # Aggregate metrics — naive scenario, for comparison only
                total_expected_naive = naive_savings_each.sum()
                roi_naive = total_expected_naive / total_cost if total_cost > 0 else np.nan

                results.append({
                    'saved_loss': saved,
                    'action_cost': cost,
                    'k_percent': k,
                    'effectiveness': effectiveness,
                    'uptake_rate': uptake_rate,
                    'n_contacted': top_n,
                    'total_expected_saving': total_expected,
                    'avg_expected_per_contact': avg_per_contact,
                    'ROI': roi,
                    'total_expected_saving_naive': total_expected_naive,
                    'ROI_naive': roi_naive,
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

    effectiveness = df['effectiveness'].iloc[0]
    uptake_rate = df['uptake_rate'].iloc[0]
    print(f"\nAssumptions used — effectiveness={effectiveness:.2f}, "
          f"uptake_rate={uptake_rate:.2f} (illustrative placeholders; "
          f"calibrate against real pilot/actuarial data before relying on "
          f"these numbers for a budget decision).")

    # Print sample results as pivot table — realistic (assumption-adjusted) scenario
    print('\nSample results (Total Expected Saving — realistic scenario, '
          'with effectiveness/uptake discount applied):')
    pivot_table = df.pivot_table(
        index=['saved_loss', 'action_cost'], 
        columns='k_percent', 
        values='total_expected_saving'
    )
    print(pivot_table)

    # Print the naive scenario alongside it so the size of the discount is visible
    print('\nFor comparison — Total Expected Saving (NAIVE scenario: assumes '
          'every intervention works and every customer accepts it; NOT a '
          'credible standalone estimate, shown only to size the effect of '
          'the effectiveness/uptake assumptions above):')
    pivot_table_naive = df.pivot_table(
        index=['saved_loss', 'action_cost'],
        columns='k_percent',
        values='total_expected_saving_naive'
    )
    print(pivot_table_naive)


def create_heatmaps(df: pd.DataFrame, k_percent_list: List[float]) -> None:
    """
    Create heatmaps for total expected saving at each percentile.

    Titles include the effectiveness/uptake assumptions used, so a reader
    looking at a saved PNG in isolation (outside the console log) still
    sees that these are assumption-adjusted, not raw, numbers.
    
    Args:
        df: DataFrame with business impact results
        k_percent_list: List of top percentiles to visualize
    """
    heatmap_dir = 'model_results/business_impact_heatmaps'
    os.makedirs(heatmap_dir, exist_ok=True)

    effectiveness = df['effectiveness'].iloc[0]
    uptake_rate = df['uptake_rate'].iloc[0]
    
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
        plt.title(f"Total Expected Saving (Top {k*100:.1f}% Targeted)\n"
                  f"Assumes effectiveness={effectiveness:.2f}, uptake_rate={uptake_rate:.2f} "
                  f"(illustrative, not fitted)",
                  fontsize=11, fontweight='bold')
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

    # ------------------------------------------------------------------
    # ILLUSTRATIVE PLACEHOLDER ASSUMPTIONS — NOT FITTED TO REAL DATA.
    # These two knobs keep the simulation from implicitly assuming a
    # "perfect world" (every intervention works, every customer accepts
    # it). The values below are reasonable-sounding defaults, nothing
    # more. Before using this simulation's dollar figures for an actual
    # budget or staffing decision, replace them with numbers from a real
    # pilot program, actuarial loss-prevention studies, or a vendor's
    # track record — and ideally re-run this as a sensitivity sweep over
    # a range of plausible values rather than a single point estimate.
    # ------------------------------------------------------------------
    EFFECTIVENESS = 0.25   # fraction of claims actually preventable once acted on
    UPTAKE_RATE = 0.70     # fraction of contacted customers who accept the action
    
    try:
        # Load data
        print("Loading predictions...")
        probs = load_predictions(PRED_PATH)
        
        # Load test IDs (optional)
        test_ids = load_test_ids(TEST_CSV_PATH)
        print(f"Loaded {len(probs)} predictions and {len(test_ids)} test IDs")
        
        # Calculate business impact
        print("\nCalculating business impact metrics...")
        print(f"Using effectiveness={EFFECTIVENESS}, uptake_rate={UPTAKE_RATE} "
              f"(illustrative placeholders — see comments in main() before "
              f"treating these numbers as real).")
        results_df = calculate_business_impact(
            probs, SAVED_LOSSES, ACTION_COSTS, K_PERCENT_LIST,
            effectiveness=EFFECTIVENESS, uptake_rate=UPTAKE_RATE
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
