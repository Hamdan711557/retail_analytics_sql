"""
integration.py - Temporal Integration Module
Combines forecasting results with pattern mining to generate bundle scores,
growth rates, and actionable business insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Growth Rate Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_product_growth(cleaned_df: pd.DataFrame) -> dict:
    """
    Computes growth rate for each product by comparing recent vs. older revenue.

    Growth formula: growth = (recent - old) / old

    The dataset is split into two equal halves:
    - older_half: first 50% of dates
    - recent_half: last 50% of dates

    Returns a dict mapping ProductName -> growth_rate
    """
    df = cleaned_df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    midpoint = df["InvoiceDate"].quantile(0.5)

    old_df = df[df["InvoiceDate"] <= midpoint]
    recent_df = df[df["InvoiceDate"] > midpoint]

    old_rev = old_df.groupby("ProductName")["TotalAmount"].sum()
    recent_rev = recent_df.groupby("ProductName")["TotalAmount"].sum()

    # Union of all products
    all_products = old_rev.index.union(recent_rev.index)
    growth_rates = {}

    for product in all_products:
        old = old_rev.get(product, 0)
        recent = recent_rev.get(product, 0)
        if old == 0:
            growth_rates[product] = 1.0  # New product, assume 100% growth
        else:
            growth_rates[product] = (recent - old) / old

    return growth_rates


# ─────────────────────────────────────────────────────────────────────────────
# Bundle Score Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_bundle_scores(rules: pd.DataFrame, growth_rates: dict) -> pd.DataFrame:
    """
    Computes a composite bundle score for each association rule.

    Score formula:
        score = (confidence * 0.6 + min(lift, 3) / 3 * 0.4) * (1 + growth)

    Args:
        rules:        Association rules DataFrame (from pattern_mining)
        growth_rates: Dict of product -> growth_rate

    Returns:
        DataFrame with bundle scores, sorted descending
    """
    if rules.empty:
        return pd.DataFrame()

    scored = rules.copy()

    # Re-extract confidence and lift as fractions (0–1)
    scored["conf_norm"] = scored["Confidence (%)"] / 100
    scored["lift_norm"] = scored["Lift"].clip(upper=3) / 3

    # Get average growth rate of consequent products
    scored["growth"] = scored["They Also Buy"].apply(
        lambda prod_str: _avg_growth(prod_str, growth_rates)
    )

    # Bundle score formula
    scored["Bundle Score"] = (
        (scored["conf_norm"] * 0.6 + scored["lift_norm"] * 0.4) *
        (1 + scored["growth"])
    ).round(4)

    # Add market trend label based on growth
    scored["Market Trend"] = scored["growth"].apply(_classify_trend)

    # Sort by bundle score
    scored = scored.sort_values("Bundle Score", ascending=False).reset_index(drop=True)

    # Final columns
    output = scored[[
        "If Customer Buys", "They Also Buy",
        "Support (%)", "Confidence (%)", "Lift",
        "growth", "Bundle Score", "Market Trend"
    ]].copy()
    output["Growth Rate (%)"] = (output["growth"] * 100).round(2)
    output.drop(columns=["growth"], inplace=True)

    return output


def _avg_growth(prod_str: str, growth_rates: dict) -> float:
    """Returns the average growth rate for a comma-separated list of products."""
    products = [p.strip() for p in prod_str.split(",")]
    rates = [growth_rates.get(p, 0) for p in products]
    return np.mean(rates) if rates else 0


def _classify_trend(growth: float) -> str:
    """Classifies growth rate as a market trend label."""
    if growth > 0.10:
        return "🟢 Growing"
    elif growth < -0.10:
        return "🔴 Declining"
    else:
        return "🟡 Stable"


# ─────────────────────────────────────────────────────────────────────────────
# Business Insights Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_business_insights(
    cleaned_df: pd.DataFrame,
    forecast_kpis: dict,
    bundle_scores: pd.DataFrame,
    growth_rates: dict
) -> list:
    """
    Generates a list of human-readable business insights.
    Combines data from all modules into actionable recommendations.
    """
    insights = []

    # ── Revenue Insight ───────────────────────────────────────────────────────
    total_rev = cleaned_df["TotalAmount"].sum()
    insights.append({
        "icon": "💰",
        "title": "Revenue Overview",
        "text": (
            f"Total historical revenue is ₹{total_rev:,.2f}. "
            f"The model forecasts ₹{forecast_kpis.get('forecast_total_6m', 0):,.2f} "
            f"over the next 6 months — a daily average of "
            f"₹{forecast_kpis.get('forecast_daily_avg', 0):,.2f}."
        ),
        "type": "info"
    })

    # ── Peak Day Insight ──────────────────────────────────────────────────────
    insights.append({
        "icon": "📅",
        "title": "Peak Revenue Day",
        "text": (
            f"The model predicts the highest revenue day will be "
            f"{forecast_kpis.get('forecast_peak_day', 'N/A')} "
            f"with an estimated ₹{forecast_kpis.get('forecast_peak_value', 0):,.2f}."
        ),
        "type": "success"
    })

    # ── Market Trend Insight ──────────────────────────────────────────────────
    trend = forecast_kpis.get("trend_direction", "Stable")
    emoji = {"Growing": "📈", "Declining": "📉", "Stable": "➡️"}.get(trend, "➡️")
    insights.append({
        "icon": emoji,
        "title": "Overall Market Trend",
        "text": (
            f"The overall sales trend is currently **{trend}**. "
            + ("Consider expanding inventory and marketing spend." if trend == "Growing"
               else "Investigate customer retention strategies." if trend == "Declining"
               else "Maintain current operations and look for new growth areas.")
        ),
        "type": "warning" if trend == "Declining" else "success" if trend == "Growing" else "info"
    })

    # ── Top Growing Products ──────────────────────────────────────────────────
    top_growing = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_growing:
        names = ", ".join([f"**{p}** ({r*100:.1f}%)" for p, r in top_growing])
        insights.append({
            "icon": "🚀",
            "title": "Fastest Growing Products",
            "text": f"Top growing products by revenue trend: {names}. Prioritize stocking these items.",
            "type": "success"
        })

    # ── Bundle Recommendation ─────────────────────────────────────────────────
    if not bundle_scores.empty:
        top_bundle = bundle_scores.iloc[0]
        insights.append({
            "icon": "🎁",
            "title": "Top Bundle Recommendation",
            "text": (
                f"Customers who buy **{top_bundle['If Customer Buys']}** frequently also purchase "
                f"**{top_bundle['They Also Buy']}** (Score: {top_bundle['Bundle Score']:.3f}). "
                f"Consider creating a bundled offer for these products."
            ),
            "type": "success"
        })

    # ── Declining Products Warning ────────────────────────────────────────────
    declining = [(p, r) for p, r in growth_rates.items() if r < -0.15]
    if declining:
        names = ", ".join([p for p, _ in declining[:3]])
        insights.append({
            "icon": "⚠️",
            "title": "Products Needing Attention",
            "text": (
                f"The following products show declining revenue trends: **{names}**. "
                f"Consider discounts, promotions, or reviewing their pricing strategy."
            ),
            "type": "warning"
        })

    # ── Top Product Insight ───────────────────────────────────────────────────
    top_products = (
        cleaned_df.groupby("ProductName")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    if not top_products.empty:
        names = ", ".join([f"**{p}**" for p in top_products.index])
        insights.append({
            "icon": "⭐",
            "title": "Top Revenue Drivers",
            "text": (
                f"Your highest revenue-generating products are {names}. "
                f"Ensure these are always in stock and well-promoted."
            ),
            "type": "info"
        })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Overall Market Trend Score
# ─────────────────────────────────────────────────────────────────────────────

def get_market_summary(growth_rates: dict) -> dict:
    """
    Computes an overall market summary from individual product growth rates.
    """
    if not growth_rates:
        return {"trend": "Unknown", "avg_growth": 0, "growing": 0, "declining": 0, "stable": 0}

    rates = list(growth_rates.values())
    avg = np.mean(rates)
    growing = sum(1 for r in rates if r > 0.10)
    declining = sum(1 for r in rates if r < -0.10)
    stable = len(rates) - growing - declining

    if avg > 0.10:
        trend = "🟢 Growing"
    elif avg < -0.10:
        trend = "🔴 Declining"
    else:
        trend = "🟡 Stable"

    return {
        "trend": trend,
        "avg_growth": round(avg * 100, 2),
        "growing": growing,
        "declining": declining,
        "stable": stable
    }
