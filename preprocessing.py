"""
preprocessing.py - Data Preprocessing Module
Cleans, validates, and transforms uploaded CSV data for downstream analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Required columns for the system
REQUIRED_COLUMNS = ["InvoiceNo", "InvoiceDate", "ProductName", "TotalAmount"]


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> dict:
    """
    Checks if the uploaded DataFrame contains all required columns.
    Returns {'valid': True} or {'valid': False, 'missing': [...]}
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return {"valid": False, "missing": missing}
    return {"valid": True}


# ─────────────────────────────────────────────────────────────────────────────
# Core Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline:
    1. Keep only required columns
    2. Remove duplicates
    3. Handle missing values
    4. Parse/normalize dates
    5. Ensure numeric TotalAmount
    6. Generate Prophet input (ds, y)
    7. Generate basket matrix (one-hot encoded)

    Returns a dict with all processed outputs and a summary log.
    """
    log = []
    original_rows = len(df)

    # ── Step 1: Keep required columns ────────────────────────────────────────
    df = df[REQUIRED_COLUMNS].copy()
    log.append(f"✅ Loaded {original_rows} rows with required columns.")

    # ── Step 2: Remove duplicate rows ────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    log.append(f"🧹 Removed {removed} duplicate rows.")

    # ── Step 3: Handle missing values ────────────────────────────────────────
    before = len(df)
    df.dropna(subset=["InvoiceNo", "InvoiceDate", "ProductName", "TotalAmount"], inplace=True)
    removed = before - len(df)
    log.append(f"🔧 Dropped {removed} rows with missing critical values.")

    # ── Step 4: Parse InvoiceDate ─────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True, errors="coerce")
    invalid_dates = df["InvoiceDate"].isna().sum()
    df.dropna(subset=["InvoiceDate"], inplace=True)
    log.append(f"📅 Parsed dates. Dropped {invalid_dates} rows with unparseable dates.")

    # ── Step 5: Normalize TotalAmount ─────────────────────────────────────────
    df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")
    df.dropna(subset=["TotalAmount"], inplace=True)
    df = df[df["TotalAmount"] > 0]  # Remove zero/negative amounts
    log.append(f"💰 TotalAmount cleaned. {len(df)} valid rows remain.")

    # ── Step 6: Sort by date ──────────────────────────────────────────────────
    df.sort_values("InvoiceDate", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Step 7: Build Prophet input (daily aggregation) ───────────────────────
    prophet_df = (
        df.groupby(df["InvoiceDate"].dt.date)["TotalAmount"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDate": "ds", "TotalAmount": "y"})
    )
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    log.append(f"📈 Prophet input prepared: {len(prophet_df)} daily data points.")

    # ── Step 8: Build Basket Matrix (one-hot encoded) ─────────────────────────
    basket_matrix = _build_basket_matrix(df)
    log.append(f"🧺 Basket matrix built: {basket_matrix.shape[0]} transactions, {basket_matrix.shape[1]} products.")

    log.append(f"✅ Preprocessing complete. Final dataset: {len(df)} rows.")

    return {
        "cleaned_df": df,
        "prophet_df": prophet_df,
        "basket_matrix": basket_matrix,
        "log": log,
        "stats": {
            "original_rows": original_rows,
            "final_rows": len(df),
            "date_range": {
                "start": str(df["InvoiceDate"].min().date()),
                "end": str(df["InvoiceDate"].max().date())
            },
            "unique_products": df["ProductName"].nunique(),
            "unique_invoices": df["InvoiceNo"].nunique(),
            "total_revenue": round(df["TotalAmount"].sum(), 2)
        }
    }


def _build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a one-hot encoded basket matrix.
    Rows = Invoices, Columns = Products, Values = 1/0
    """
    basket = (
        df.groupby(["InvoiceNo", "ProductName"])["TotalAmount"]
        .sum()
        .unstack(fill_value=0)
    )
    # Convert to binary (bought / not bought)
    basket = (basket > 0).astype(int)
    return basket


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Generate Sample CSV
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_csv() -> pd.DataFrame:
    """
    Generates a realistic sample dataset for demo/testing purposes.
    Returns a DataFrame with the required columns.
    """
    np.random.seed(42)
    products = [
        "Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
        "USB Hub", "Webcam", "Desk Lamp", "Notebook", "Pen Set",
        "Phone Stand", "Cable Organizer", "Mousepad", "Speaker", "Charger"
    ]
    dates = pd.date_range(start="2023-01-01", end="2024-06-30", freq="D")
    n = 2000

    data = {
        "InvoiceNo": [f"INV{1000 + i}" for i in range(n)],
        "InvoiceDate": np.random.choice(dates, n),
        "ProductName": np.random.choice(products, n, p=[
            0.12, 0.10, 0.09, 0.08, 0.08,
            0.07, 0.07, 0.06, 0.06, 0.06,
            0.05, 0.05, 0.05, 0.05, 0.01
        ]),
        "TotalAmount": np.round(np.random.uniform(15, 800, n), 2)
    }
    return pd.DataFrame(data)
