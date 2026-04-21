"""
forecasting.py - Sales Forecasting Module
Uses Facebook Prophet to generate 6-month sales forecasts with trend decomposition.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Main Forecasting Function
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast(prophet_df: pd.DataFrame, periods: int = 180) -> dict:
    """
    Trains a Prophet model and generates a 6-month forecast.

    Args:
        prophet_df: DataFrame with columns 'ds' (date) and 'y' (revenue)
        periods:    Number of days to forecast (default 180 ≈ 6 months)

    Returns:
        dict with forecast DataFrame, model, and computed insights
    """
    # ── Train Prophet Model ───────────────────────────────────────────────────
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,    # Controls trend flexibility
        seasonality_prior_scale=10.0,    # Controls seasonality strength
        interval_width=0.95              # 95% confidence interval
    )

    model.fit(prophet_df)

    # ── Generate Future Dates ─────────────────────────────────────────────────
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)

    # ── Monthly Predictions ───────────────────────────────────────────────────
    monthly = _aggregate_monthly(forecast, prophet_df)

    # ── Product-Level Insights (based on historical data) ────────────────────
    # Note: Prophet forecasts aggregate sales; product insights come from cleaned_df
    # These are calculated in integration.py using the full dataset.

    # ── Weekly Trend Summary ─────────────────────────────────────────────────
    weekly_trend = _compute_weekly_trend(forecast, prophet_df)

    # ── KPI Summary ──────────────────────────────────────────────────────────
    future_only = forecast[forecast["ds"] > prophet_df["ds"].max()]
    kpis = {
        "forecast_total_6m": round(future_only["yhat"].sum(), 2),
        "forecast_daily_avg": round(future_only["yhat"].mean(), 2),
        "forecast_peak_day": future_only.loc[future_only["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d"),
        "forecast_peak_value": round(future_only["yhat"].max(), 2),
        "historical_daily_avg": round(prophet_df["y"].mean(), 2),
        "trend_direction": _get_trend_direction(forecast)
    }

    return {
        "model": model,
        "forecast": forecast,
        "monthly_predictions": monthly,
        "weekly_trend": weekly_trend,
        "kpis": kpis,
        "prophet_df": prophet_df
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_monthly(forecast: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily forecast to monthly totals.
    Marks rows as 'Historical' or 'Forecast'.
    """
    last_hist_date = historical["ds"].max()

    monthly = (
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .copy()
    )
    monthly["Month"] = monthly["ds"].dt.to_period("M")
    monthly = (
        monthly.groupby("Month")
        .agg(
            Revenue=("yhat", "sum"),
            Lower=("yhat_lower", "sum"),
            Upper=("yhat_upper", "sum")
        )
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].astype(str)
    monthly["Type"] = monthly.apply(
        lambda r: "Historical" if pd.Period(r["Month"]) <= last_hist_date.to_period("M") else "Forecast",
        axis=1
    )
    monthly["Revenue"] = monthly["Revenue"].round(2)
    monthly["Lower"] = monthly["Lower"].round(2)
    monthly["Upper"] = monthly["Upper"].round(2)
    return monthly


def _compute_weekly_trend(forecast: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes average revenue by day-of-week for historical period.
    """
    # Merge forecast with historical to get actual values
    merged = forecast.merge(historical, on="ds", how="left")
    # Use actual where available, predicted otherwise
    merged["value"] = merged["y"].fillna(merged["yhat"])
    merged["DayOfWeek"] = merged["ds"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = (
        merged.groupby("DayOfWeek")["value"]
        .mean()
        .reindex(day_order)
        .reset_index()
        .rename(columns={"value": "AvgRevenue"})
    )
    weekly["AvgRevenue"] = weekly["AvgRevenue"].round(2)
    return weekly


def _get_trend_direction(forecast: pd.DataFrame) -> str:
    """
    Determines overall trend by comparing first and last trend values.
    Returns 'Growing', 'Stable', or 'Declining'.
    """
    trend = forecast["trend"]
    start = trend.iloc[:30].mean()
    end = trend.iloc[-30:].mean()
    change_pct = (end - start) / (start + 1e-9) * 100

    if change_pct > 5:
        return "Growing"
    elif change_pct < -5:
        return "Declining"
    else:
        return "Stable"


# ─────────────────────────────────────────────────────────────────────────────
# Product-Level Forecast Insights
# ─────────────────────────────────────────────────────────────────────────────

def product_level_insights(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates revenue and sales count per product from historical data.
    Returns a sorted DataFrame with product insights.
    """
    insights = (
        cleaned_df.groupby("ProductName")
        .agg(
            TotalRevenue=("TotalAmount", "sum"),
            OrderCount=("InvoiceNo", "nunique"),
            AvgOrderValue=("TotalAmount", "mean")
        )
        .reset_index()
        .sort_values("TotalRevenue", ascending=False)
        .reset_index(drop=True)
    )
    insights["TotalRevenue"] = insights["TotalRevenue"].round(2)
    insights["AvgOrderValue"] = insights["AvgOrderValue"].round(2)
    insights["RevenueShare%"] = (
        insights["TotalRevenue"] / insights["TotalRevenue"].sum() * 100
    ).round(2)
    return insights
