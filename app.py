"""
GRU Energy Audit Analyzer
=========================
Pre-survey report tool for energy auditors.
Upload a customer meter file (and optionally AMI data) to generate
a complete pre-survey briefing before an on-site visit.

Run:
    streamlit run app.py
"""

import io
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GRU Energy Audit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F7F7F5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1C1C1E;
        border-right: 1px solid #2C2C2E;
    }
    [data-testid="stSidebar"] * {
        color: #E5E5E7 !important;
    }
    [data-testid="stSidebar"] .stFileUploader label {
        color: #98989D !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Header bar */
    .audit-header {
        background: #1C1C1E;
        color: #F5F5F7;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: baseline;
        gap: 1rem;
    }
    .audit-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0;
        color: #F5F5F7;
    }
    .audit-header .subtitle {
        font-size: 0.8rem;
        color: #98989D;
        letter-spacing: 0.03em;
    }

    /* Customer info banner */
    .customer-banner {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-left: 4px solid #1C1C1E;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
    }
    .customer-banner .customer-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1C1C1E;
        margin-bottom: 0.15rem;
    }
    .customer-banner .customer-meta {
        font-size: 0.8rem;
        color: #6E6E73;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Stat cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .stat-card .stat-label {
        font-size: 0.7rem;
        color: #98989D;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.4rem;
    }
    .stat-card .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #1C1C1E;
        line-height: 1.2;
    }
    .stat-card .stat-unit {
        font-size: 0.75rem;
        color: #6E6E73;
        margin-top: 0.2rem;
    }
    .stat-card.warning {
        border-left: 3px solid #FF6B35;
    }
    .stat-card.ok {
        border-left: 3px solid #34C759;
    }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #98989D;
        border-bottom: 1px solid #E5E5EA;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Division tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #EBEBEB;
        border-radius: 6px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 0.4rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: #6E6E73;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF;
        color: #1C1C1E !important;
        font-weight: 600;
    }

    /* Anomaly badge */
    .anomaly-badge {
        display: inline-block;
        background: #FF3B30;
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.1rem 0.5rem;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
    }
    .ok-badge {
        display: inline-block;
        background: #34C759;
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.1rem 0.5rem;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Checklist items */
    .checklist-item {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 5px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        font-size: 0.875rem;
        color: #1C1C1E;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .checklist-item.priority {
        border-left: 3px solid #FF6B35;
        background: #FFF8F5;
    }
    .checklist-icon {
        font-size: 0.9rem;
        flex-shrink: 0;
        margin-top: 0.05rem;
    }

    /* Correlation badge */
    .corr-strong { color: #FF3B30; font-weight: 600; }
    .corr-moderate { color: #FF9500; font-weight: 600; }
    .corr-weak { color: #34C759; font-weight: 600; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Upload zone */
    [data-testid="stFileUploadDropzone"] {
        background: #2C2C2E !important;
        border: 1px dashed #48484A !important;
        border-radius: 6px !important;
    }

    /* Divider */
    hr {border: none; border-top: 1px solid #E5E5EA; margin: 1rem 0;}

    /* Matplotlib figures */
    .stImage > img { border-radius: 6px; }
</style>
, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    LATITUDE         = 29.6516
    LONGITUDE        = -82.3248
    COMFORT_BASELINE = 65
    FIGURE_DPI       = 110
    COLORS = {
        "electric" : "#2E86AB",
        "water"    : "#028090",
        "gas"      : "#F18F01",
        "anomaly"  : "#FF3B30",
        "normal"   : "#C7C7CC",
        "rolling"  : "#1C1C1E",
        "hot"      : "#E63946",
        "cold"     : "#457B9D",
        "mild"     : "#2A9D8F",
        "ami"      : "#6A4C93",
    }
    DIV_COLORS = {
        "Electricity" : "#2E86AB",
        "Water"       : "#028090",
        "Gas"         : "#F18F01",
    }


# ─────────────────────────────────────────────────────────────────────────────
# METER LOADER
# ─────────────────────────────────────────────────────────────────────────────

class MeterLoader:
    COLUMN_MAP = {
        "Division"   : "division",
        "Device"     : "device",
        "MR Reason"  : "mr_reason",
        "MR Type"    : "mr_type",
        "MR Date"    : "mr_date",
        "Days"       : "days",
        "MR Result"  : "mr_result",
        "MR Unit"    : "mr_unit",
        "Consumption": "consumption",
        "Avg."       : "avg_daily",
        "Avg"        : "avg_daily",
    }
    NON_READ_REASONS = {3}
    VLINE_REASONS    = {6, 21, 22}

    def __init__(self, fileobj):
        self.fileobj       = fileobj
        self.df            = None
        self.has_mr_reason = False

    def _find_sheet(self, xl):
        for name in xl.sheet_names:
            if "consumption" in name.lower():
                return name
        raise ValueError(
            f"No consumption sheet found. Sheets: {xl.sheet_names}"
        )

    def _detect_header_row(self, xl, sheet):
        for i in range(5):
            df = pd.read_excel(xl, sheet_name=sheet, header=i, nrows=1)
            df.columns = [str(c).strip() for c in df.columns]
            if "Division" in df.columns:
                return i
        return 0

    def load(self):
        xl         = pd.ExcelFile(self.fileobj)
        sheet      = self._find_sheet(xl)
        header_row = self._detect_header_row(xl, sheet)

        df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=self.COLUMN_MAP)

        self.has_mr_reason = "mr_reason" in df.columns
        df["mr_date"]      = pd.to_datetime(df["mr_date"], errors="coerce")

        if df["consumption"].dtype == object:
            df["consumption"] = df["consumption"].astype(str).str.replace(",", "", regex=False)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

        for col in ["mr_result", "days", "avg_daily"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["mr_date"])

        if self.has_mr_reason:
            df["mr_reason"] = pd.to_numeric(df["mr_reason"], errors="coerce")
            df = df[~df["mr_reason"].isin(self.NON_READ_REASONS)]
            df = df[(df["consumption"] > 0) | (df["mr_reason"].isin(self.VLINE_REASONS))]
        else:
            df = df[df["consumption"] > 0]

        df = df[df["days"] > 0]
        df = df.sort_values(["division", "device", "mr_date"]).reset_index(drop=True)
        self.df = df
        return df

    def get_division(self, name):
        if self.df is None:
            raise RuntimeError("Call load() first")
        sub = self.df[self.df["division"] == name].copy()
        if not sub.empty:
            sub = sub[sub["mr_date"] > sub["mr_date"].min()].reset_index(drop=True)
        return sub

    def get_customer_info(self):
        try:
            ms       = pd.read_excel(self.fileobj, sheet_name="Master Sheet", header=None)
            cell_0_6 = str(ms.iloc[0, 6]).strip() if pd.notna(ms.iloc[0, 6]) else ""
            offset   = 1 if cell_0_6 and not any(c.isdigit() for c in cell_0_6) else 0

            def safe(r, c):
                try:
                    v = ms.iloc[r + offset, c]
                    return str(v).strip() if pd.notna(v) else ""
                except Exception:
                    return ""

            # Find address by scanning for "Address" label
            address, city = "", ""
            for r in range(len(ms)):
                for c in range(len(ms.columns)):
                    if str(ms.iloc[r, c]).strip().lower() == "address":
                        address = str(ms.iloc[r, c + 1]).strip() if pd.notna(ms.iloc[r, c + 1]) else ""
                        city    = str(ms.iloc[r + 1, c + 1]).strip() if pd.notna(ms.iloc[r + 1, c + 1]) else ""
                        break

            return {
                "account"  : safe(0, 6),
                "name"     : safe(1, 6),
                "own_rent" : safe(2, 6),
                "community": safe(3, 6),
                "address"  : address,
                "city"     : city or "Gainesville FL",
            }
        except Exception:
            return {"account": "—", "name": "—", "own_rent": "—", "community": "—", "address": "—", "city": "—"}


# ─────────────────────────────────────────────────────────────────────────────
# METER FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class MeterFeatures:
    def __init__(self, df):
        self.df = df.copy().sort_values("mr_date").reset_index(drop=True)

    def compute(self):
        df             = self.df
        total          = df["consumption"].sum()
        total_days     = df["days"].sum()
        daily_avg      = total / total_days if total_days > 0 else None
        peak           = df["consumption"].max()
        base           = df["consumption"].quantile(0.05)
        avg_interval   = df["days"].mean()
        period_series  = df.set_index("mr_date")["consumption"]
        rolling_avg    = period_series.rolling(window=3).mean()

        iso_cols = [c for c in ["consumption", "days", "avg_daily"] if c in df.columns]
        iso_data = df[iso_cols].dropna()
        df["anomaly"] = False
        if len(iso_data) >= 5:
            preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(iso_data)
            df.loc[iso_data.index, "anomaly"] = (preds == -1)

        unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""

        return {
            "total"        : total,
            "daily_avg"    : daily_avg,
            "peak"         : peak,
            "base"         : base,
            "avg_interval" : avg_interval,
            "n_reads"      : len(df),
            "n_anomalies"  : int(df["anomaly"].sum()),
            "unit"         : unit,
            "period_series": period_series,
            "rolling_avg"  : rolling_avg,
            "df"           : df,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AMI LOADER
# ─────────────────────────────────────────────────────────────────────────────

class AMILoader:
    SHEET_MAP = {
        "ELECTRIC": "Electric", "Electric": "Electric",
        "Sheet1"  : "Electric", "SHEET1"  : "Electric",
        "WATER"   : "Water",    "Water"   : "Water",
        "GAS"     : "Gas",      "Gas"     : "Gas",
    }
    UNITS = {"Electric": "kWh", "Water": "Gal", "Gas": "CCF"}

    def __init__(self, fileobj):
        self.fileobj   = fileobj
        self.df        = None
        self.util_type = None
        self.unit      = None

    def load(self):
        xl    = pd.ExcelFile(self.fileobj)
        sheet = next((n for n in xl.sheet_names if n in self.SHEET_MAP), xl.sheet_names[0])

        self.util_type = self.SHEET_MAP.get(sheet, "Electric")
        self.unit      = self.UNITS[self.util_type]

        df = pd.read_excel(xl, sheet_name=sheet, header=None, skiprows=4)
        df = df[[0, 1]].copy()
        df.columns = ["timestamp", "raw_value"]

        df["timestamp"] = (df["timestamp"].astype(str)
                           .str.replace(r"\s+E[SD]T.*$", "", regex=True)
                           .str.strip())
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%b %d, %Y - %I:%M %p", errors="coerce")

        df["value"] = (df["raw_value"].astype(str)
                       .str.replace(",", "", regex=False)
                       .str.extract(r"([\d.]+)")[0])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        if self.util_type == "Electric":
            df["value"] = df["value"] / 1000

        df["kwh"] = df["value"]
        df = df.dropna(subset=["timestamp", "value"])
        df = df[df["value"] > 0]
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        return df


class AMIFeatures:
    def __init__(self, df):
        self.df = df.copy()

    def compute(self):
        df           = self.df.sort_values("timestamp")
        deltas       = df["timestamp"].diff().dropna()
        interval_min = int(deltas.mode()[0].total_seconds() / 60)
        base_kwh     = df["kwh"].quantile(0.05)
        base_kw      = base_kwh / (interval_min / 60)
        peak_kwh     = df["kwh"].max()
        peak_kw      = peak_kwh / (interval_min / 60)
        df["date"]   = df["timestamp"].dt.date
        daily        = df.groupby("date")["kwh"].sum()
        df["hour"]   = df["timestamp"].dt.hour
        hourly       = df.groupby("hour")["kwh"].mean()

        return {
            "interval_min" : interval_min,
            "base_kwh"     : base_kwh,
            "base_kw"      : base_kw,
            "peak_kwh"     : peak_kwh,
            "peak_kw"      : peak_kw,
            "daily_avg"    : daily.mean(),
            "daily_series" : daily,
            "peak_day"     : pd.Timestamp(daily.idxmax()),
            "hourly"       : hourly,
            "df"           : df,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_temperature(start_str: str, end_str: str) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude"        : Config.LATITUDE,
                "longitude"       : Config.LONGITUDE,
                "start_date"      : start_str,
                "end_date"        : end_str,
                "daily"           : "temperature_2m_max,temperature_2m_min",
                "temperature_unit": "fahrenheit",
                "timezone"        : "America/New_York",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["daily"]
        df   = pd.DataFrame({
            "date"    : pd.to_datetime(data["time"]),
            "temp_max": data["temperature_2m_max"],
            "temp_min": data["temperature_2m_min"],
        })
        df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
        return df.set_index("date")
    except Exception:
        return None


def merge_temp(df_div, df_temp):
    df   = df_div[df_div["consumption"] > 0].copy().sort_values("mr_date").reset_index(drop=True)
    avgs = []
    for _, row in df.iterrows():
        end   = row["mr_date"]
        start = end - pd.Timedelta(days=int(row["days"]))
        mask  = (df_temp.index >= start) & (df_temp.index <= end)
        avgs.append(df_temp[mask]["temp_avg"].mean() if not df_temp[mask].empty else None)
    df["temp_avg"]   = avgs
    df["temp_delta"] = (df["temp_avg"] - Config.COMFORT_BASELINE).abs()
    return df.dropna(subset=["temp_avg"])


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS  (return matplotlib figures for st.pyplot)
# ─────────────────────────────────────────────────────────────────────────────

def fig_consumption(feats, division, customer_name):
    df     = feats["df"]
    unit   = feats["unit"]
    color  = Config.DIV_COLORS.get(division, "#2E86AB")
    normal = df[~df["anomaly"]]
    anom   = df[df["anomaly"]]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(normal["mr_date"], normal["consumption"], width=20,
           color=color, alpha=0.55, label="Normal")
    ax.bar(anom["mr_date"], anom["consumption"], width=20,
           color=Config.COLORS["anomaly"], alpha=0.9, label="Anomaly")
    ax.plot(feats["rolling_avg"].index, feats["rolling_avg"].values,
            color=Config.COLORS["rolling"], linewidth=1.8,
            linestyle="--", label="3-Period Avg", zorder=5)

    ax.set_title(f"{division} — Consumption History", fontsize=10,
                 fontweight="bold", color="#1C1C1E", pad=10)
    ax.set_ylabel(unit, fontsize=8, color="#6E6E73")
    ax.tick_params(labelsize=7, colors="#6E6E73")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E5E5EA")
    ax.legend(fontsize=7, frameon=False)
    fig.autofmt_xdate()
    plt.tight_layout(pad=0.8)
    return fig


def fig_temp_scatter(df_merged, division, customer_name):
    df   = df_merged.copy()
    unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""
    df["daily"] = df["consumption"] / df["days"]

    if division == "Gas":
        x_col  = "temp_avg"
        xlabel = "Avg Temperature (°F)"
        r      = df["daily"].corr(df["temp_avg"])
        r_type = "Linear r"
    else:
        x_col  = "temp_delta"
        xlabel = "|Temp − 65°F|  (deviation from comfort zone)"
        r      = df["daily"].corr(df["temp_delta"])
        r_type = "V-shape r"

    def tc(t):
        if t >= 80: return Config.COLORS["hot"]
        if t <= 55: return Config.COLORS["cold"]
        return Config.COLORS["mild"]

    colors = [tc(t) for t in df["temp_avg"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(df[x_col], df["daily"], c=colors, s=55, alpha=0.85, edgecolors="white", zorder=3)
    z = np.polyfit(df[x_col], df["daily"], 1)
    xl = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), color="#1C1C1E", linewidth=1.5, linestyle="--")

    ax.set_title(f"{division} vs Temperature  ({r_type} = {r:.2f})",
                 fontsize=9, fontweight="bold", color="#1C1C1E", pad=8)
    ax.set_xlabel(xlabel, fontsize=8, color="#6E6E73")
    ax.set_ylabel(f"{unit}/day", fontsize=8, color="#6E6E73")
    ax.tick_params(labelsize=7, colors="#6E6E73")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E5E5EA")

    handles = [
        mpatches.Patch(color=Config.COLORS["hot"],  label="Hot (>80°F)"),
        mpatches.Patch(color=Config.COLORS["cold"], label="Cold (<55°F)"),
        mpatches.Patch(color=Config.COLORS["mild"], label="Mild"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    plt.tight_layout(pad=0.8)
    return fig, r


def fig_ami_hourly(ami_feats, unit, customer_name):
    hourly    = ami_feats["hourly"]
    base      = ami_feats["base_kwh"]
    peak_hour = int(hourly.idxmax())

    colors = [Config.COLORS["anomaly"] if h == peak_hour
              else Config.COLORS["ami"] for h in hourly.index]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(hourly.index, hourly.values, color=colors, alpha=0.85, width=0.7)
    ax.axhline(base, color=Config.COLORS["mild"], linewidth=1.8, linestyle="--",
               label=f"Base Load  ({ami_feats['base_kw']:.2f} kW)")
    ax.set_title("Average Load by Hour of Day", fontsize=10,
                 fontweight="bold", color="#1C1C1E", pad=10)
    ax.set_xlabel("Hour  (0 = midnight)", fontsize=8, color="#6E6E73")
    ax.set_ylabel(f"{unit}/interval", fontsize=8, color="#6E6E73")
    ax.set_xticks(range(0, 24))
    ax.tick_params(labelsize=7, colors="#6E6E73")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E5E5EA")
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout(pad=0.8)
    return fig


def fig_ami_daily(ami_feats, unit, customer_name):
    daily    = ami_feats["daily_series"]
    avg      = ami_feats["daily_avg"]
    peak_day = ami_feats["peak_day"].date()
    colors   = ["#E63946" if d == peak_day else Config.COLORS["ami"]
                for d in daily.index]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(daily.index, daily.values, color=colors, alpha=0.85)
    ax.axhline(avg, color=Config.COLORS["gas"], linewidth=1.8, linestyle="--",
               label=f"Daily Avg  ({avg:.1f} {unit})")
    ax.set_title(f"Daily Total Usage  —  peak day: {peak_day}",
                 fontsize=10, fontweight="bold", color="#1C1C1E", pad=10)
    ax.set_ylabel(f"{unit}/day", fontsize=8, color="#6E6E73")
    ax.tick_params(labelsize=7, colors="#6E6E73")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E5E5EA")
    ax.legend(fontsize=7, frameon=False)
    fig.autofmt_xdate()
    plt.tight_layout(pad=0.8)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def stat_card(label, value, unit="", warning=False, ok=False):
    cls = "stat-card"
    if warning: cls += " warning"
    elif ok:    cls += " ok"
    st.markdown(f"""
    <div class="{cls}">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-unit">{unit}</div>
    </div>
    , unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def checklist_item(text, priority=False):
    cls  = "checklist-item priority" if priority else "checklist-item"
    icon = "⚑" if priority else "□"
    st.markdown(f"""
    <div class="{cls}">
        <span class="checklist-icon">{icon}</span>
        <span>{text}</span>
    </div>
    , unsafe_allow_html=True)


def corr_label(r, division):
    abs_r = abs(r)
    if division == "Gas":
        if r < -0.6:   return "strong", "Strong heating dependency"
        if r < -0.3:   return "moderate", "Moderate heating relationship"
        return "weak", "Weak weather correlation"
    else:
        if abs_r > 0.6: return "strong", "Strong HVAC dependency"
        if abs_r > 0.3: return "moderate", "Moderate weather sensitivity"
        return "weak", "Weak weather correlation"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.15em; text-transform:uppercase; color:#636366;
                    margin-bottom:0.4rem;">GRU Energy Audit</div>
        <div style="font-size:1.1rem; font-weight:600; color:#F5F5F7;">
            Pre-Survey Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:#636366; margin-bottom:0.5rem;">Customer File</div>', unsafe_allow_html=True)
    meter_file = st.file_uploader("meter", type=["xlsx"], label_visibility="collapsed")

    st.markdown('<div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:#636366; margin: 1rem 0 0.5rem 0;">AMI File <span style="color:#48484A">(optional)</span></div>', unsafe_allow_html=True)
    ami_file = st.file_uploader("ami", type=["xlsx"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2C2C2E; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#636366; line-height:1.6;">
        <div style="margin-bottom:0.5rem; color:#98989D; font-weight:500;">HOW TO USE</div>
        1. Upload customer .xlsx file<br>
        2. Optionally add AMI data<br>
        3. Review analysis tabs<br>
        4. Print checklist before survey
    </div>
    , unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
<div class="audit-header">
    <h1>⚡ GRU Energy Audit</h1>
    <span class="subtitle">Pre-Survey Analysis Tool</span>
</div>
, unsafe_allow_html=True)

if meter_file is None:
    st.markdown(
    <div style="text-align:center; padding: 4rem 2rem; color:#98989D;">
        <div style="font-size:2.5rem; margin-bottom:1rem;">📂</div>
        <div style="font-size:1rem; font-weight:500; color:#6E6E73; margin-bottom:0.5rem;">
            Upload a customer file to begin
        </div>
        <div style="font-size:0.8rem; color:#98989D;">
            Use the sidebar to upload a meter reading Excel file
        </div>
    </div>
    , unsafe_allow_html=True)
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading meter data…"):
    try:
        loader   = MeterLoader(meter_file)
        df_all   = loader.load()
        customer = loader.get_customer_info()
    except Exception as e:
        st.error(f"Could not load file: {e}")
        st.stop()

# ── Customer banner ──────────────────────────────────────────────────────────
divisions_found = df_all["division"].unique().tolist()
div_str         = "  ·  ".join(divisions_found)
acct            = customer.get("account", "—")
own_rent        = customer.get("own_rent", "—")
community       = customer.get("community", "—")

st.markdown(f"""
<div class="customer-banner">
    <div class="customer-name">{customer.get('name', '—')}</div>
    <div class="customer-meta">
        {customer.get('address', '—')}&nbsp;&nbsp;·&nbsp;&nbsp;
        {customer.get('city', '')}
    </div>
    <div class="customer-meta" style="margin-top:0.3rem;">
        Acct: {acct}&nbsp;&nbsp;·&nbsp;&nbsp;
        {own_rent}&nbsp;&nbsp;·&nbsp;&nbsp;
        {community}&nbsp;&nbsp;·&nbsp;&nbsp;
        Divisions: {div_str}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Fetch temperature ────────────────────────────────────────────────────────
with st.spinner("Fetching weather data…"):
    start_str = (df_all["mr_date"].min() - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    end_str   = df_all["mr_date"].max().strftime("%Y-%m-%d")
    df_temp   = fetch_temperature(start_str, end_str)

temp_status = "✔ Weather data loaded" if df_temp is not None else "⚠ Weather data unavailable"

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_labels = ["Meter Analysis"]
if ami_file: tab_labels.append("AMI Analysis")
tab_labels.append("Pre-Survey Checklist")

tabs = st.tabs(tab_labels)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — METER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    div_order = ["Electricity", "Water", "Gas"]
    divs      = [d for d in div_order if d in divisions_found]

    if not divs:
        st.warning("No division data found in this file.")
    else:
        div_tabs = st.tabs(divs)

        for div_tab, div_name in zip(div_tabs, divs):
            with div_tab:
                df_div = loader.get_division(div_name)
                if df_div.empty:
                    st.info(f"No {div_name} data found.")
                    continue

                feats = MeterFeatures(df_div).compute()
                unit  = feats["unit"]
                color = Config.DIV_COLORS.get(div_name, "#2E86AB")

                # ── Stats row ────────────────────────────────────────────
                section_header("Summary Statistics")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    stat_card("Daily Average",
                              f"{feats['daily_avg']:.1f}" if feats["daily_avg"] else "—",
                              f"{unit}/day",
                              warning=(div_name == "Electricity" and feats["daily_avg"] and feats["daily_avg"] > 40))
                with c2:
                    stat_card("Total Consumption",
                              f"{feats['total']:,.0f}", unit)
                with c3:
                    stat_card("Peak Period",
                              f"{feats['peak']:,.0f}", unit,
                              warning=(feats["peak"] > feats["daily_avg"] * 45 if feats["daily_avg"] else False))
                with c4:
                    stat_card("Base Load (P5)",
                              f"{feats['base']:,.0f}", unit)
                with c5:
                    stat_card("Anomalies",
                              feats["n_anomalies"],
                              f"of {feats['n_reads']} reads",
                              warning=(feats["n_anomalies"] > 0),
                              ok=(feats["n_anomalies"] == 0))

                # ── Anomaly detail ────────────────────────────────────────
                if feats["n_anomalies"] > 0:
                    anom_df = feats["df"][feats["df"]["anomaly"]][["mr_date", "consumption", "days", "avg_daily"]].copy()
                    anom_df["mr_date"] = anom_df["mr_date"].dt.strftime("%Y-%m-%d")
                    anom_df.columns    = ["Date", "Consumption", "Days", "Daily Avg"]
                    st.markdown(f'<span class="anomaly-badge">⚠ {feats["n_anomalies"]} ANOMALY PERIOD{"S" if feats["n_anomalies"] > 1 else ""}</span>', unsafe_allow_html=True)
                    st.dataframe(anom_df.reset_index(drop=True), use_container_width=True, hide_index=True)
                else:
                    st.markdown('<span class="ok-badge">✔ No anomalies detected</span>', unsafe_allow_html=True)

                # ── Consumption chart ─────────────────────────────────────
                section_header("Consumption History")
                st.pyplot(fig_consumption(feats, div_name, customer.get("name", "")))

                # ── Temperature correlation ───────────────────────────────
                if df_temp is not None:
                    section_header("Temperature Correlation")
                    df_merged = merge_temp(df_div, df_temp)
                    if not df_merged.empty:
                        fig_tc, r = fig_temp_scatter(df_merged, div_name, customer.get("name", ""))
                        strength, label = corr_label(r, div_name)

                        col_chart, col_interp = st.columns([2, 1])
                        with col_chart:
                            st.pyplot(fig_tc)
                        with col_interp:
                            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                            r_type = "Linear r" if div_name == "Gas" else "V-shape r"
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">{r_type}</div>
                                <div class="stat-value corr-{strength}">{r:.2f}</div>
                                <div class="stat-unit">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            if div_name == "Gas":
                                interp = (
                                    "Gas usage rises strongly as temps drop. "
                                    "Space heating is the primary driver — inspect furnace/heat pump."
                                    if r < -0.5 else
                                    "Some heating relationship present. "
                                    "Gas likely used for multiple purposes (water heater, cooking)."
                                )
                            elif abs(r) > 0.5:
                                interp = (
                                    "Usage spikes in both summer and winter, "
                                    "indicating HVAC is the dominant load. "
                                    "Check AC efficiency and heating system."
                                )
                            else:
                                interp = (
                                    "Usage is relatively stable across temperatures. "
                                    "Non-HVAC loads (appliances, water heater, lighting) "
                                    "are likely driving consumption."
                                )
                            st.markdown(f"""
                            <div style="font-size:0.78rem; color:#6E6E73; margin-top:0.75rem;
                                        line-height:1.6; padding:0.75rem; background:#F7F7F5;
                                        border-radius:5px; border:1px solid #E5E5EA;">
                                {interp}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Not enough temperature overlap to compute correlation.")
                else:
                    st.caption(f"⚠ {temp_status} — temperature correlation unavailable.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — AMI ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

if ami_file:
    with tabs[1]:
        with st.spinner("Loading AMI data…"):
            try:
                ami_loader = AMILoader(ami_file)
                df_ami     = ami_loader.load()
                ami_feats  = AMIFeatures(df_ami).compute()
                unit       = ami_loader.unit
                util_type  = ami_loader.util_type
            except Exception as e:
                st.error(f"Could not load AMI file: {e}")
                st.stop()

        # ── AMI header info ───────────────────────────────────────────────
        date_min = df_ami["timestamp"].min().strftime("%Y-%m-%d")
        date_max = df_ami["timestamp"].max().strftime("%Y-%m-%d")
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#6E6E73; margin-bottom:1rem;">
            <b style="color:#1C1C1E">{util_type}</b> &nbsp;·&nbsp;
            {ami_feats['interval_min']}-minute intervals &nbsp;·&nbsp;
            {len(df_ami):,} records &nbsp;·&nbsp;
            {date_min} → {date_max}
        </div>
        """, unsafe_allow_html=True)

        # ── AMI stat cards ────────────────────────────────────────────────
        section_header("Demand Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_card("Base Load",
                      f"{ami_feats['base_kw']:.3f}", "kW")
        with c2:
            stat_card("Peak Demand",
                      f"{ami_feats['peak_kw']:.3f}", "kW",
                      warning=(ami_feats["peak_kw"] > ami_feats["base_kw"] * 10))
        with c3:
            stat_card("Daily Average",
                      f"{ami_feats['daily_avg']:.2f}", f"{unit}/day")
        with c4:
            stat_card("Peak Day",
                      ami_feats["peak_day"].strftime("%b %d, %Y"), "")

        # ── Peak hour insight ─────────────────────────────────────────────
        hourly    = ami_feats["hourly"]
        peak_hour = int(hourly.idxmax())
        off_peak  = float(hourly.loc[0:6].mean())
        on_peak   = float(hourly.loc[14:19].mean())

        section_header("Usage Pattern")
        p1, p2 = st.columns(2)
        with p1:
            stat_card("Peak Hour", f"{peak_hour}:00",
                      f"{hourly[peak_hour]:.3f} {unit}/interval")
        with p2:
            ratio = on_peak / off_peak if off_peak > 0 else 0
            pattern = (
                "High daytime usage" if on_peak > off_peak * 2 else
                "Nighttime-heavy"    if off_peak > on_peak else
                "Relatively flat"
            )
            stat_card("Load Pattern", pattern,
                      f"on-peak {on_peak:.3f} / off-peak {off_peak:.3f} {unit}",
                      warning=(on_peak > off_peak * 2))

        # ── Hourly chart ──────────────────────────────────────────────────
        section_header("Hourly Load Profile  (red bar = peak hour)")
        st.pyplot(fig_ami_hourly(ami_feats, unit, customer.get("name", "")))

        # ── Daily chart ───────────────────────────────────────────────────
        section_header("Daily Totals  (red bar = peak day)")
        st.pyplot(fig_ami_daily(ami_feats, unit, customer.get("name", "")))


# ═════════════════════════════════════════════════════════════════════════════
# TAB — PRE-SURVEY CHECKLIST
# ═════════════════════════════════════════════════════════════════════════════

with tabs[-1]:
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#98989D; margin-bottom:1.5rem;">
        Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} &nbsp;·&nbsp;
        {customer.get('name','—')} &nbsp;·&nbsp; {customer.get('address','—')}
    </div>
    """, unsafe_allow_html=True)

    # ── Build dynamic checklist based on analysis ─────────────────────────
    priority_items = []
    standard_items = []

    df_elec  = loader.get_division("Electricity")
    df_water = loader.get_division("Water")
    df_gas   = loader.get_division("Gas")

    # Electricity checks
    if not df_elec.empty:
        ef = MeterFeatures(df_elec).compute()
        if ef["daily_avg"] and ef["daily_avg"] > 40:
            priority_items.append(f"High electricity usage ({ef['daily_avg']:.1f} kWh/day) — inspect HVAC system, water heater, insulation")
        if ef["n_anomalies"] > 2:
            priority_items.append(f"{ef['n_anomalies']} anomaly periods detected — ask about equipment changes or occupancy shifts")
        elif ef["n_anomalies"] > 0:
            standard_items.append(f"{ef['n_anomalies']} anomaly period detected — confirm with customer")

        if df_temp is not None:
            dm = merge_temp(df_elec, df_temp)
            if not dm.empty:
                dm["daily"] = dm["consumption"] / dm["days"]
                r_elec = dm["daily"].corr(dm["temp_delta"])
                if abs(r_elec) > 0.6:
                    priority_items.append(f"Strong HVAC dependency (r = {r_elec:.2f}) — check thermostat settings, duct leaks, AC efficiency")
                elif abs(r_elec) > 0.3:
                    standard_items.append(f"Moderate weather sensitivity (r = {r_elec:.2f}) — verify HVAC maintenance schedule")

    # Water checks
    if not df_water.empty:
        wf = MeterFeatures(df_water).compute()
        if wf["daily_avg"] and wf["daily_avg"] > 150:
            priority_items.append(f"High water usage ({wf['daily_avg']:.1f} gal/day) — check for irrigation system, leaks, water heater")
        if wf["n_anomalies"] > 0:
            standard_items.append(f"Water anomaly periods detected ({wf['n_anomalies']}) — ask about landscape irrigation or pool")

    # Gas checks
    if not df_gas.empty:
        gf = MeterFeatures(df_gas).compute()
        standard_items.append("Gas service present — inspect water heater type, range/oven, furnace or heat strip")
        if df_temp is not None:
            dm = merge_temp(df_gas, df_temp)
            if not dm.empty:
                dm["daily"] = dm["consumption"] / dm["days"]
                r_gas = dm["daily"].corr(dm["temp_avg"])
                if r_gas < -0.6:
                    priority_items.append(f"Strong gas heating dependency (r = {r_gas:.2f}) — inspect furnace, insulation, duct system")

    # AMI checks
    if ami_file and "ami_feats" in dir():
        if ami_feats["peak_kw"] > ami_feats["base_kw"] * 10:
            priority_items.append(f"High peak-to-base ratio ({ami_feats['peak_kw']:.1f}x base) — look for cycling equipment or intermittent large loads")
        off_p = float(ami_feats["hourly"].loc[0:6].mean())
        on_p  = float(ami_feats["hourly"].loc[14:19].mean())
        if off_p > on_p * 1.5:
            standard_items.append("Nighttime-heavy load pattern — check always-on equipment, EV charger, water heater schedule")

    # Own/rent note
    if customer.get("own_rent", "").lower() == "renter":
        standard_items.append("Renter — discuss landlord approval process for any equipment upgrades")

    # Standard items always present
    standard_items.extend([
        "Verify thermostat type (programmable / smart / manual) and current settings",
        "Check air filter condition and last replacement date",
        "Inspect windows and doors for air leaks (especially older units)",
        "Document appliance ages and approximate conditions",
        "Review lighting throughout unit (LED conversion opportunities)",
        "Ask about occupancy hours and work-from-home schedule",
    ])

    # ── Render checklist ──────────────────────────────────────────────────
    if priority_items:
        section_header(f"Priority Items  ({len(priority_items)})")
        for item in priority_items:
            checklist_item(item, priority=True)

    section_header(f"Standard Checks  ({len(standard_items)})")
    for item in standard_items:
        checklist_item(item)

    # ── Quick stats summary for printout ─────────────────────────────────
    section_header("Quick Reference")

    ref_cols = st.columns(len(divs) if divs else 1)
    for col, div_name in zip(ref_cols, [d for d in ["Electricity","Water","Gas"] if d in divisions_found]):
        df_d = loader.get_division(div_name)
        if not df_d.empty:
            f    = MeterFeatures(df_d).compute()
            with col:
                st.markdown(f
                <div class="stat-card">
                    <div class="stat-label">{div_name}</div>
                    <div style="font-size:0.82rem; color:#1C1C1E; line-height:2; margin-top:0.25rem;">
                        <span style="color:#98989D;">Daily avg</span>&nbsp;
                        <b>{f['daily_avg']:.1f if f['daily_avg'] else '—'} {f['unit']}/day</b><br>
                        <span style="color:#98989D;">Peak</span>&nbsp;
                        <b>{f['peak']:,.0f} {f['unit']}</b><br>
                        <span style="color:#98989D;">Reads</span>&nbsp;
                        <b>{f['n_reads']}</b><br>
                        <span style="color:#98989D;">Anomalies</span>&nbsp;
                        <b style="color:{'#FF3B30' if f['n_anomalies'] > 0 else '#34C759'}">{f['n_anomalies']}</b>
                    </div>
                </div>
                , unsafe_allow_html=True)

    st.markdown(
    <div style="margin-top:1.5rem; font-size:0.72rem; color:#C7C7CC; text-align:center;">
        GRU Energy Audit Analyzer &nbsp;·&nbsp; Internal use only
    </div>
    , unsafe_allow_html=True
