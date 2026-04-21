# 📊 Integrated Retail Analytics — SQLite Edition

**Student:** Hamdan Rasheed V H  
**Project Type:** Final Year Project  
**Database:** SQLite (no server required!)

---

## 🚀 Overview

A full-stack retail analytics web application featuring:
- **Sales Forecasting** using Facebook Prophet (6-month predictions)
- **Market Basket Analysis** using FP-Growth algorithm
- **Temporal Integration** — bundle scores, growth rates, business insights
- **Secure Login / Register** using SQLite + bcrypt (no MongoDB needed)

---

## 📁 File Structure

```
retail_analytics_sql/
│
├── dashboard.py         ← Main Streamlit app (UI + routing)
├── preprocessing.py     ← Module 1: Data cleaning & transformation
├── forecasting.py       ← Module 2: Prophet-based forecasting
├── pattern_mining.py    ← Module 3: FP-Growth pattern mining
├── integration.py       ← Module 4: Growth rates, bundle scores, insights
├── database.py          ← SQLite connection & all SQL operations
├── auth.py              ← Login, Register, bcrypt hashing, session
├── requirements.txt     ← Python dependencies (no pymongo!)
├── README.md            ← This file
└── .streamlit/
    └── config.toml      ← Streamlit dark theme config
```

> **Note:** `retail_analytics.db` (SQLite file) is created automatically
> in this folder on first run. You don't need to create it manually.

---

## ⚙️ Local Setup

### 1. Install Python 3.10+

Download from https://python.org

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run dashboard.py
```

Opens at **http://localhost:8501**

The SQLite database (`retail_analytics.db`) is created automatically on first run.

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - SQLite retail analytics"
git remote add origin https://github.com/YOUR_USERNAME/retail-analytics.git
git push -u origin main
```

### Step 2: Deploy

1. Go to https://share.streamlit.io
2. Click **New App**
3. Connect your GitHub repo
4. Set **Main file path:** `dashboard.py`
5. Click **Deploy** — no secrets needed for SQLite!

> **Note for Streamlit Cloud:** SQLite data resets when the app restarts
> (ephemeral filesystem). For persistent user data on cloud, consider
> upgrading to PostgreSQL using the same SQL queries with `psycopg2`.

---

## 📊 How to Use

| Step | Where | Action |
|------|-------|--------|
| 1 | Auth page | Register a new account or sign in |
| 2 | Upload Data tab | Upload CSV or click "Use sample dataset" |
| 3 | Upload Data tab | Click **Run Preprocessing Pipeline** |
| 4 | Sales Forecast tab | Click **Run Sales Forecast** |
| 5 | Patterns tab | Click **Run FP-Growth Algorithm** |
| 6 | Insights tab | View auto-generated business insights |

---

## 🗄️ Database Schema (SQLite)

### Table: `users`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment user ID |
| username | TEXT UNIQUE | Login username |
| email | TEXT UNIQUE | User email address |
| password | TEXT | bcrypt hashed password |
| created_at | TEXT | ISO timestamp of registration |
| last_login | TEXT | ISO timestamp of last login |

### Table: `uploaded_files`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment record ID |
| user_id | INTEGER FK | References users.id |
| filename | TEXT | Name of uploaded file |
| row_count | INTEGER | Number of rows in uploaded file |
| uploaded_at | TEXT | ISO timestamp |

---

## 📋 Required CSV Format

| Column | Type | Example |
|--------|------|---------|
| InvoiceNo | String | INV1001 |
| InvoiceDate | Date | 2024-01-15 |
| ProductName | String | Laptop |
| TotalAmount | Number | 599.99 |

---

## 🧩 Module Descriptions

| Module | File | Algorithm |
|--------|------|-----------|
| Preprocessing | preprocessing.py | Pandas / NumPy |
| Sales Forecasting | forecasting.py | Facebook Prophet |
| Pattern Mining | pattern_mining.py | FP-Growth (mlxtend) |
| Integration | integration.py | Custom growth + bundle scoring |
| Database | database.py | SQLite3 |
| Authentication | auth.py | bcrypt |

### Bundle Score Formula
```
score = (confidence × 0.6 + min(lift, 3)/3 × 0.4) × (1 + growth_rate)
```

### Growth Rate Formula
```
growth = (recent_revenue − old_revenue) / old_revenue
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Database | SQLite3 (built into Python) |
| Auth | bcrypt |
| Forecasting | Facebook Prophet |
| Pattern Mining | mlxtend FP-Growth |
| Visualization | Plotly |
| Data | Pandas, NumPy |

---

*Final Year Project — Hamdan Rasheed V H*  
*SQLite Edition — No external database server required*
