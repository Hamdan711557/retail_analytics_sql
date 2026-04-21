"""
pattern_mining.py - Market Basket / Pattern Mining Module
Uses FP-Growth algorithm (mlxtend) to discover frequent itemsets and association rules.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pattern Mining Function
# ─────────────────────────────────────────────────────────────────────────────

def run_pattern_mining(basket_matrix: pd.DataFrame,
                       min_support: float = 0.02,
                       min_confidence: float = 0.1,
                       min_lift: float = 1.0) -> dict:
    """
    Runs FP-Growth on the basket matrix to find frequent itemsets
    and generate association rules.

    Args:
        basket_matrix:   One-hot encoded DataFrame (rows=invoices, cols=products)
        min_support:     Minimum support threshold (default 2%)
        min_confidence:  Minimum confidence threshold (default 10%)
        min_lift:        Minimum lift threshold for filtering rules (default 1.0)

    Returns:
        dict with frequent_itemsets, association_rules DataFrames, and summary stats
    """
    if basket_matrix.empty or basket_matrix.shape[0] < 5:
        return {
            "frequent_itemsets": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "summary": {"error": "Not enough data for pattern mining."}
        }

    # ── Run FP-Growth ─────────────────────────────────────────────────────────
    # Ensure binary values
    basket_bool = basket_matrix.astype(bool)

    frequent_itemsets = fpgrowth(
        basket_bool,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return {
            "frequent_itemsets": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "summary": {"error": "No frequent itemsets found. Try lowering min_support."}
        }

    # ── Generate Association Rules ────────────────────────────────────────────
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Filter by lift
    rules = rules[rules["lift"] >= min_lift].copy()

    # ── Clean and Format ──────────────────────────────────────────────────────
    rules = _format_rules(rules)
    frequent_itemsets = _format_itemsets(frequent_itemsets)

    # ── Summary Stats ─────────────────────────────────────────────────────────
    summary = _build_summary(frequent_itemsets, rules)

    return {
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "summary": summary
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Cleans up association rules for display."""
    if rules.empty:
        return rules

    rules = rules.copy()

    # Convert frozensets to readable strings
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    # Round metrics
    rules["support"] = rules["support"].round(4)
    rules["confidence"] = rules["confidence"].round(4)
    rules["lift"] = rules["lift"].round(4)

    # Add support percentage
    rules["support_pct"] = (rules["support"] * 100).round(2)
    rules["confidence_pct"] = (rules["confidence"] * 100).round(2)

    # Sort by lift descending
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Select and rename columns
    display_cols = ["antecedents", "consequents", "support_pct", "confidence_pct", "lift"]
    rules = rules[display_cols].rename(columns={
        "antecedents": "If Customer Buys",
        "consequents": "They Also Buy",
        "support_pct": "Support (%)",
        "confidence_pct": "Confidence (%)",
        "lift": "Lift"
    })

    return rules


def _format_itemsets(itemsets: pd.DataFrame) -> pd.DataFrame:
    """Cleans up frequent itemsets for display."""
    itemsets = itemsets.copy()
    itemsets["itemsets"] = itemsets["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    itemsets["itemset_size"] = itemsets["itemsets"].apply(lambda x: len(x.split(", ")))
    itemsets["support_pct"] = (itemsets["support"] * 100).round(2)
    itemsets = itemsets.sort_values("support", ascending=False).reset_index(drop=True)
    itemsets = itemsets[["itemsets", "itemset_size", "support_pct"]].rename(columns={
        "itemsets": "Product Combination",
        "itemset_size": "Size",
        "support_pct": "Support (%)"
    })
    return itemsets


def _build_summary(itemsets: pd.DataFrame, rules: pd.DataFrame) -> dict:
    """Builds a summary statistics dictionary."""
    summary = {
        "total_itemsets": len(itemsets),
        "total_rules": len(rules),
        "avg_support": 0,
        "avg_confidence": 0,
        "avg_lift": 0,
        "top_rule": None,
        "best_lift": 0
    }

    if not rules.empty:
        summary["avg_support"] = round(rules["Support (%)"].mean(), 2)
        summary["avg_confidence"] = round(rules["Confidence (%)"].mean(), 2)
        summary["avg_lift"] = round(rules["Lift"].mean(), 4)
        top = rules.iloc[0]
        summary["top_rule"] = f"{top['If Customer Buys']} → {top['They Also Buy']}"
        summary["best_lift"] = round(rules["Lift"].max(), 4)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Top Product Pairs
# ─────────────────────────────────────────────────────────────────────────────

def get_top_pairs(rules: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Returns the top N product pairs by lift."""
    if rules.empty:
        return pd.DataFrame()
    return rules.head(n)
