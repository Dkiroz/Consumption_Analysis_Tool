"""
GRU Energy Audit Analyzer — Pre-Survey Report Tool
===================================================

A command-line tool for energy auditors to analyze customer utility data
before conducting on-site surveys.

Usage:
    python gru_audit.py <customer_file.xlsx> [--ami <ami_file.xlsx>] [--save]

Features:
    - Meter reading history analysis (Electric, Water, Gas)
    - Weather-normalized consumption analysis
    - Anomaly detection with IsolationForest
    - Temperature correlation (V-shape for HVAC, linear for Gas)
    - AMI interval data analysis (15-min load profiles)
    - Pre-survey summary report

Author: GRU Energy Audit Team
Version: 2.0
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration."""
    
    # Gainesville, FL coordinates
    LATITUDE = 29.6516
    LONGITUDE = -82.3248
    
    # Anomaly detection
    ISOLATION_CONTAMINATION = 0.05
    
    # Temperature correlation
    COMFORT_BASELINE = 65  # °F
    
    # Display
    FIGURE_DPI = 100
    DEFAULT_FIGSIZE = (12, 4)
    
    # Colors
    COLORS = {
        "electric": "#2E86AB",
        "water": "#028090",
        "gas": "#F18F01",
        "anomaly": "#C73E1D",
        "normal": "#A3B18A",
        "hot": "#E63946",
        "cold": "#457B9D",
        "mild": "#2A9D8F",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METER DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class MeterLoader:
    """
    Load and clean GRU meter reading Excel files.
    
    Handles format variations:
        - Extra title rows above headers
        - MR Reason codes vs MR Type descriptions
        - Comma-separated numbers
        - Various sheet name capitalizations
    """
    
    COLUMN_MAP = {
        "Division": "division",
        "Device": "device",
        "MR Reason": "mr_reason",
        "MR Type": "mr_type",
        "MR Date": "mr_date",
        "Days": "days",
        "MR Result": "mr_result",
        "MR Unit": "mr_unit",
        "Consumption": "consumption",
        "Avg.": "avg_daily",
        "Avg": "avg_daily",
    }
    
    NON_READ_REASONS = {3}  # Estimated reads
    VLINE_REASONS = {6, 21, 22}  # Move-in, meter change
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.has_mr_reason = False
    
    def _find_sheet(self, xl: pd.ExcelFile) -> str:
        """Find the consumption sheet."""
        for name in xl.sheet_names:
            if "consumption" in name.lower():
                # Verify sheet has data
                df = pd.read_excel(xl, sheet_name=name, header=None, nrows=5)
                if not df.empty and len(df.columns) > 0:
                    return name
        raise ValueError(
            f"No valid consumption data found. "
            f"Sheets available: {xl.sheet_names}. "
            f"Please ensure 'Consumption History' tab has meter reading data."
        )
    
    def _detect_header_row(self, xl: pd.ExcelFile, sheet: str) -> int:
        """Find row containing 'Division' header."""
        for i in range(5):
            df = pd.read_excel(xl, sheet_name=sheet, header=i, nrows=1)
            # Convert columns to strings and strip whitespace
            df.columns = [str(c).strip() for c in df.columns]
            if "Division" in df.columns:
                return i
        return 0
    
    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        """Clean numeric column (handle commas, strings)."""
        if series.dtype == object:
            s = series.astype(str).str.replace(",", "", regex=False)
            return pd.to_numeric(s, errors="coerce")
        return pd.to_numeric(series, errors="coerce")
    
    def load(self) -> pd.DataFrame:
        """Load and clean meter reading data."""
        xl = pd.ExcelFile(self.filepath)
        sheet = self._find_sheet(xl)
        header_row = self._detect_header_row(xl, sheet)
        
        df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
        # Convert columns to strings and strip whitespace
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=self.COLUMN_MAP)
        
        self.has_mr_reason = "mr_reason" in df.columns
        
        # Parse dates
        df["mr_date"] = pd.to_datetime(df["mr_date"], errors="coerce")
        
        # Clean numeric columns
        df["consumption"] = self._clean_numeric(df["consumption"])
        for col in ["mr_result", "days", "avg_daily"]:
            if col in df.columns:
                df[col] = self._clean_numeric(df[col])
        
        # Drop invalid rows
        df = df.dropna(subset=["mr_date"])
        
        # Filter based on MR Reason
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
    
    def get_division(self, name: str) -> pd.DataFrame:
        """Get data for a specific division."""
        if self.df is None:
            raise RuntimeError("Call load() first")
        sub = self.df[self.df["division"] == name].copy()
        if not sub.empty:
            sub = sub[sub["mr_date"] > sub["mr_date"].min()].reset_index(drop=True)
        return sub
    
    def get_customer_info(self) -> Dict[str, str]:
        """Extract customer info from Master Sheet."""
        try:
            ms = pd.read_excel(self.filepath, sheet_name="Master Sheet", header=None)
            
            # Detect row offset
            cell_0_6 = str(ms.iloc[0, 6]).strip() if pd.notna(ms.iloc[0, 6]) else ""
            offset = 1 if cell_0_6 and not any(c.isdigit() for c in cell_0_6) else 0
            
            def safe_get(r, c):
                try:
                    val = ms.iloc[r + offset, c]
                    return str(val).strip() if pd.notna(val) else ""
                except:
                    return ""
            
            # Get address from row 4, col 6
            address = safe_get(4, 6) if offset == 0 else safe_get(3, 6)
            city = safe_get(5, 6) if offset == 0 else safe_get(4, 6)
            
            return {
                "account": safe_get(0, 6),
                "name": safe_get(1, 6),
                "own_rent": safe_get(2, 6),
                "community": safe_get(3, 6),
                "address": address,
                "city": city if city else "Gainesville FL",
            }
        except Exception:
            return {"account": "Unknown", "name": "Unknown"}


# ═══════════════════════════════════════════════════════════════════════════════
# METER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class MeterFeatures:
    """Compute features and detect anomalies from meter data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_values("mr_date").reset_index(drop=True)
    
    def compute(self) -> Dict[str, Any]:
        """Compute all features."""
        df = self.df
        
        total_consumption = df["consumption"].sum()
        total_days = df["days"].sum()
        daily_avg = total_consumption / total_days if total_days > 0 else None
        peak = df["consumption"].max()
        base = df["consumption"].quantile(0.05)
        avg_interval = df["days"].mean()
        
        # Rolling averages
        period_series = df.set_index("mr_date")["consumption"]
        rolling_avg = period_series.rolling(window=3).mean()
        
        # Anomaly detection
        iso_cols = [c for c in ["consumption", "days", "avg_daily"] if c in df.columns]
        iso_data = df[iso_cols].dropna()
        df["anomaly"] = False
        
        if len(iso_data) >= 5:
            model = IsolationForest(
                contamination=Config.ISOLATION_CONTAMINATION,
                random_state=42
            )
            preds = model.fit_predict(iso_data)
            df.loc[iso_data.index, "anomaly"] = (preds == -1)
        
        n_anomalies = int(df["anomaly"].sum())
        unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""
        
        # Data quality score
        quality = self._quality_score(df)
        
        return {
            "total_consumption": total_consumption,
            "daily_avg": daily_avg,
            "peak": peak,
            "base": base,
            "avg_interval": avg_interval,
            "n_reads": len(df),
            "n_anomalies": n_anomalies,
            "unit": unit,
            "quality_score": quality,
            "period_series": period_series,
            "rolling_avg": rolling_avg,
            "df": df,
        }
    
    def _quality_score(self, df: pd.DataFrame) -> int:
        """Compute data quality score (0-100)."""
        score = 100
        
        if df["consumption"].isna().any():
            score -= 10
        if df["days"].std() > 10:
            score -= 5
        if len(df) < 12:
            score -= 15
        if (df["consumption"] == 0).sum() > 2:
            score -= 10
        if len(df) > 1:
            gaps = df["mr_date"].diff().dt.days
            if gaps.max() > 60:
                score -= 20
        
        return max(0, score)


# ═══════════════════════════════════════════════════════════════════════════════
# AMI LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class AMILoader:
    """Load and clean AMI (15-minute interval) data."""
    
    SHEET_MAP = {
        "ELECTRIC": "Electric", "Electric": "Electric", "Sheet1": "Electric",
        "WATER": "Water", "Water": "Water",
        "GAS": "Gas", "Gas": "Gas",
    }
    
    UNITS = {"Electric": "kWh", "Water": "Gal", "Gas": "CCF"}
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.util_type = None
        self.unit = None
    
    def load(self) -> pd.DataFrame:
        """Load and clean AMI data."""
        xl = pd.ExcelFile(self.filepath)
        
        # Find sheet
        sheet = None
        for name in xl.sheet_names:
            if name in self.SHEET_MAP:
                sheet = name
                break
        sheet = sheet or xl.sheet_names[0]
        
        self.util_type = self.SHEET_MAP.get(sheet, "Electric")
        self.unit = self.UNITS[self.util_type]
        
        # Read data (skip 4 metadata rows)
        df = pd.read_excel(xl, sheet_name=sheet, header=None, skiprows=4)
        df = df[[0, 1]].copy()
        df.columns = ["timestamp", "raw_value"]
        
        # Parse timestamp
        df["timestamp"] = (df["timestamp"].astype(str)
                          .str.replace(r"\s+E[SD]T.*$", "", regex=True)
                          .str.strip())
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="%b %d, %Y - %I:%M %p",
            errors="coerce"
        )
        
        # Extract numeric value
        df["value"] = (df["raw_value"].astype(str)
                      .str.replace(",", "", regex=False)
                      .str.extract(r"([\d.]+)")[0])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Convert Wh to kWh for electric
        if self.util_type == "Electric":
            df["value"] = df["value"] / 1000
        
        df["kwh"] = df["value"]  # Legacy column name
        
        df = df.dropna(subset=["timestamp", "value"])
        df = df[df["value"] > 0]
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        self.df = df
        return df


class AMIFeatures:
    """Compute features from AMI interval data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def compute(self) -> Dict[str, Any]:
        """Compute AMI features."""
        df = self.df.sort_values("timestamp")
        
        # Interval detection
        deltas = df["timestamp"].diff().dropna()
        interval = deltas.mode()[0]
        interval_min = int(interval.total_seconds() / 60)
        
        # Base load and peak
        base_kwh = df["kwh"].quantile(0.05)
        base_kw = base_kwh / (interval_min / 60)
        peak_kwh = df["kwh"].max()
        peak_kw = peak_kwh / (interval_min / 60)
        
        # Daily aggregates
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date")["kwh"].sum()
        daily_avg = daily.mean()
        peak_day = pd.Timestamp(daily.idxmax())
        
        # Hourly profile
        df["hour"] = df["timestamp"].dt.hour
        hourly = df.groupby("hour")["kwh"].mean()
        
        return {
            "interval_min": interval_min,
            "base_kwh": base_kwh,
            "base_kw": base_kw,
            "peak_kwh": peak_kwh,
            "peak_kw": peak_kw,
            "daily_avg": daily_avg,
            "daily_series": daily,
            "peak_day": peak_day,
            "hourly_profile": hourly,
            "df": df,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_temperature(start_date, end_date) -> Optional[pd.DataFrame]:
    """Fetch temperature data from Open-Meteo API."""
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": Config.LATITUDE,
        "longitude": Config.LONGITUDE,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()["daily"]
        
        df = pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "temp_max": data["temperature_2m_max"],
            "temp_min": data["temperature_2m_min"],
        })
        df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
        df = df.set_index("date")
        return df
    except Exception as e:
        print(f"  ⚠ Could not fetch temperature data: {e}")
        return None


def merge_temp(df_div: pd.DataFrame, df_temp: pd.DataFrame) -> pd.DataFrame:
    """Merge consumption with temperature data."""
    df = df_div.copy().sort_values("mr_date").reset_index(drop=True)
    df = df[df["consumption"] > 0]
    
    temp_avgs = []
    for _, row in df.iterrows():
        end = row["mr_date"]
        start = end - pd.Timedelta(days=int(row["days"]))
        mask = (df_temp.index >= start) & (df_temp.index <= end)
        period = df_temp[mask]
        temp_avgs.append(period["temp_avg"].mean() if not period.empty else None)
    
    df["temp_avg"] = temp_avgs
    df["temp_delta"] = (df["temp_avg"] - Config.COMFORT_BASELINE).abs()
    return df.dropna(subset=["temp_avg"])


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generate pre-survey report with visualizations."""
    
    def __init__(self, customer_info: Dict, save_path: Optional[str] = None):
        self.info = customer_info
        self.save_path = save_path
        self.name = customer_info.get("name", "Unknown")
        self.account = customer_info.get("account", "Unknown")
    
    def _save_or_show(self, fig, name: str):
        """Save figure or display it."""
        if self.save_path:
            path = os.path.join(self.save_path, f"{self.account}_{name}.png")
            fig.savefig(path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
    
    def plot_consumption_history(self, feats: Dict, division: str):
        """Plot consumption history with anomalies highlighted."""
        df = feats["df"]
        unit = feats["unit"]
        
        fig, ax = plt.subplots(figsize=Config.DEFAULT_FIGSIZE)
        
        normal = df[~df["anomaly"]]
        anomaly = df[df["anomaly"]]
        
        ax.bar(normal["mr_date"], normal["consumption"], width=20,
               color=Config.COLORS["normal"], alpha=0.8, label="Normal")
        ax.bar(anomaly["mr_date"], anomaly["consumption"], width=20,
               color=Config.COLORS["anomaly"], alpha=0.9, label="Anomaly")
        
        # Rolling average
        ax.plot(feats["rolling_avg"].index, feats["rolling_avg"].values,
                color="#1D3557", linewidth=2, linestyle="--", label="3-Period Avg")
        
        ax.set_title(f"{self.name} — {division} Consumption History", fontsize=12, fontweight="bold")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        plt.tight_layout()
        
        self._save_or_show(fig, f"{division.lower()}_history")
    
    def plot_temp_correlation(self, df_merged: pd.DataFrame, division: str):
        """Plot temperature correlation scatter."""
        if df_merged.empty:
            return
        
        unit = df_merged["mr_unit"].iloc[0] if "mr_unit" in df_merged.columns else ""
        df = df_merged.copy()
        df["daily"] = df["consumption"] / df["days"]
        
        # Color by temperature
        def temp_color(t):
            if t >= 80: return Config.COLORS["hot"]
            if t <= 55: return Config.COLORS["cold"]
            return Config.COLORS["mild"]
        
        colors = [temp_color(t) for t in df["temp_avg"]]
        
        # Use temp_delta for V-shape (Electric/Water) or temp_avg for Gas
        if division == "Gas":
            x_col, xlabel = "temp_avg", "Avg Temperature (°F)"
            r = df["daily"].corr(df["temp_avg"])
        else:
            x_col, xlabel = "temp_delta", "|Temp − 65°F|"
            r = df["daily"].corr(df["temp_delta"])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[x_col], df["daily"], c=colors, s=60, alpha=0.8, edgecolors="white")
        
        # Trend line
        z = np.polyfit(df[x_col], df["daily"], 1)
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), color="#E63946", linewidth=2, linestyle="--")
        
        # Legend
        handles = [
            mpatches.Patch(color=Config.COLORS["hot"], label="Hot (>80°F)"),
            mpatches.Patch(color=Config.COLORS["cold"], label="Cold (<55°F)"),
            mpatches.Patch(color=Config.COLORS["mild"], label="Mild"),
        ]
        ax.legend(handles=handles, fontsize=8)
        
        ax.set_title(f"{self.name} — {division} vs Temperature (r={r:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{unit}/day")
        plt.tight_layout()
        
        self._save_or_show(fig, f"{division.lower()}_temp")
    
    def plot_ami_profile(self, feats: Dict, unit: str):
        """Plot AMI hourly load profile."""
        hourly = feats["hourly_profile"]
        base = feats["base_kwh"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        bars = ax.bar(hourly.index, hourly.values, color="#6A4C93", alpha=0.85, width=0.7)
        ax.axhline(base, color="#2A9D8F", linewidth=2, linestyle="--",
                   label=f"Base Load ({feats['base_kw']:.2f} kW)")
        
        # Highlight peak hours
        peak_hour = hourly.idxmax()
        bars[peak_hour].set_color("#E63946")
        
        ax.set_title(f"{self.name} — Average Load by Hour", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(f"{unit}/interval")
        ax.set_xticks(range(24))
        ax.legend(fontsize=8)
        plt.tight_layout()
        
        self._save_or_show(fig, "ami_hourly")
    
    def plot_ami_daily(self, feats: Dict, unit: str):
        """Plot AMI daily totals."""
        daily = feats["daily_series"]
        avg = feats["daily_avg"]
        peak_day = feats["peak_day"].date()
        
        colors = ["#E63946" if d == peak_day else "#2E86AB" for d in daily.index]
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(daily.index, daily.values, color=colors, alpha=0.85)
        ax.axhline(avg, color="#F18F01", linewidth=2, linestyle="--",
                   label=f"Daily Avg ({avg:.1f} {unit})")
        
        ax.set_title(f"{self.name} — Daily Usage (Peak: {peak_day})", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{unit}/day")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        plt.tight_layout()
        
        self._save_or_show(fig, "ami_daily")


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-SURVEY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(text: str, char: str = "═"):
    """Print formatted header."""
    width = 65
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(text: str):
    """Print section header."""
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")


def print_stat(label: str, value, unit: str = ""):
    """Print formatted statistic."""
    if value is None:
        print(f"  {label:.<30} N/A")
    elif isinstance(value, float):
        print(f"  {label:.<30} {value:,.2f} {unit}")
    else:
        print(f"  {label:.<30} {value} {unit}")


def generate_presurvey_report(
    meter_file: str,
    ami_file: Optional[str] = None,
    save_charts: bool = False
):
    """Generate comprehensive pre-survey report."""
    
    print_header("GRU ENERGY AUDIT — PRE-SURVEY REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Meter File: {os.path.basename(meter_file)}")
    if ami_file:
        print(f"  AMI File: {os.path.basename(ami_file)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # LOAD METER DATA
    # ═══════════════════════════════════════════════════════════════════
    
    print_section("Loading Data")
    
    loader = MeterLoader(meter_file)
    df_all = loader.load()
    customer = loader.get_customer_info()
    
    print(f"  ✔ Loaded {len(df_all)} meter readings")
    print(f"  Divisions: {df_all['division'].unique().tolist()}")
    
    # Customer info
    print_header("CUSTOMER INFORMATION", "─")
    print(f"  Name    : {customer.get('name', 'N/A')}")
    print(f"  Account : {customer.get('account', 'N/A')}")
    print(f"  Address : {customer.get('address', 'N/A')}")
    print(f"            {customer.get('city', '')}")
    print(f"  Own/Rent: {customer.get('own_rent', 'N/A')}")
    
    # Setup chart generator
    save_path = os.path.dirname(meter_file) if save_charts else None
    report = ReportGenerator(customer, save_path)
    
    # Fetch temperature data
    print_section("Fetching Weather Data")
    start = df_all["mr_date"].min() - pd.Timedelta(days=35)
    end = df_all["mr_date"].max()
    df_temp = fetch_temperature(start, end)
    if df_temp is not None:
        print(f"  ✔ Temperature data: {len(df_temp)} days")
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYZE EACH DIVISION
    # ═══════════════════════════════════════════════════════════════════
    
    divisions = {
        "Electricity": {"color": "electric", "temp_type": "V-shape"},
        "Water": {"color": "water", "temp_type": "V-shape"},
        "Gas": {"color": "gas", "temp_type": "Linear"},
    }
    
    for div_name, div_config in divisions.items():
        df_div = loader.get_division(div_name)
        
        if df_div.empty:
            continue
        
        print_header(f"{div_name.upper()} ANALYSIS", "─")
        
        # Compute features
        feats = MeterFeatures(df_div).compute()
        
        print_stat("Total Reads", feats["n_reads"])
        print_stat("Total Consumption", feats["total_consumption"], feats["unit"])
        print_stat("Daily Average", feats["daily_avg"], f"{feats['unit']}/day")
        print_stat("Peak Period", feats["peak"], feats["unit"])
        print_stat("Base Load (P5)", feats["base"], feats["unit"])
        print_stat("Avg Read Interval", feats["avg_interval"], "days")
        print_stat("Data Quality", feats["quality_score"], "/100")
        
        # Anomalies
        if feats["n_anomalies"] > 0:
            print(f"\n  ⚠ ANOMALIES DETECTED: {feats['n_anomalies']}")
            anomalies = feats["df"][feats["df"]["anomaly"]]
            for _, row in anomalies.iterrows():
                print(f"    • {row['mr_date'].strftime('%Y-%m-%d')}: "
                      f"{row['consumption']:,.0f} {feats['unit']} "
                      f"({row['avg_daily']:.1f}/day)")
        else:
            print(f"\n  ✔ No anomalies detected")
        
        # Plot consumption history
        report.plot_consumption_history(feats, div_name)
        
        # Temperature correlation
        if df_temp is not None:
            df_merged = merge_temp(df_div, df_temp)
            if not df_merged.empty:
                df_merged["daily"] = df_merged["consumption"] / df_merged["days"]
                
                if div_name == "Gas":
                    r = df_merged["daily"].corr(df_merged["temp_avg"])
                    corr_type = "Linear"
                else:
                    r = df_merged["daily"].corr(df_merged["temp_delta"])
                    corr_type = "V-shape"
                
                print(f"\n  Temperature Correlation ({corr_type}): r = {r:.2f}")
                
                if abs(r) > 0.5:
                    print(f"  → Strong HVAC dependency")
                elif abs(r) > 0.3:
                    print(f"  → Moderate weather sensitivity")
                else:
                    print(f"  → Weak weather correlation")
                
                report.plot_temp_correlation(df_merged, div_name)
    
    # ═══════════════════════════════════════════════════════════════════
    # AMI ANALYSIS (if provided)
    # ═══════════════════════════════════════════════════════════════════
    
    if ami_file and os.path.exists(ami_file):
        print_header("AMI INTERVAL ANALYSIS", "─")
        
        try:
            ami_loader = AMILoader(ami_file)
            df_ami = ami_loader.load()
            ami_feats = AMIFeatures(df_ami).compute()
            
            print(f"  Utility Type: {ami_loader.util_type}")
            print(f"  Interval: {ami_feats['interval_min']} minutes")
            print(f"  Date Range: {df_ami['timestamp'].min().strftime('%Y-%m-%d')} → "
                  f"{df_ami['timestamp'].max().strftime('%Y-%m-%d')}")
            print()
            print_stat("Base Load", ami_feats["base_kw"], "kW")
            print_stat("Peak Demand", ami_feats["peak_kw"], "kW")
            print_stat("Daily Average", ami_feats["daily_avg"], f"{ami_loader.unit}/day")
            print_stat("Peak Day", ami_feats["peak_day"].strftime("%Y-%m-%d"))
            
            # Peak hour analysis
            hourly = ami_feats["hourly_profile"]
            peak_hour = hourly.idxmax()
            off_peak = hourly.loc[0:6].mean()
            on_peak = hourly.loc[14:19].mean()
            
            print(f"\n  Peak Hour: {peak_hour}:00 ({hourly[peak_hour]:.3f} {ami_loader.unit}/interval)")
            print(f"  Off-Peak (12AM-6AM): {off_peak:.3f} {ami_loader.unit}/interval")
            print(f"  On-Peak (2PM-7PM): {on_peak:.3f} {ami_loader.unit}/interval")
            
            if on_peak > off_peak * 2:
                print(f"  → High daytime usage pattern")
            elif off_peak > on_peak:
                print(f"  → Nighttime-heavy usage pattern")
            
            # Plot AMI charts
            report.plot_ami_profile(ami_feats, ami_loader.unit)
            report.plot_ami_daily(ami_feats, ami_loader.unit)
            
        except Exception as e:
            print(f"  ⚠ Error loading AMI data: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # SURVEY RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    print_header("PRE-SURVEY CHECKLIST", "═")
    
    # Build recommendations based on analysis
    recommendations = []
    
    # Check for high electricity usage
    df_elec = loader.get_division("Electricity")
    if not df_elec.empty:
        elec_feats = MeterFeatures(df_elec).compute()
        if elec_feats["daily_avg"] and elec_feats["daily_avg"] > 40:
            recommendations.append("□ High electricity usage — inspect HVAC system, water heater, insulation")
        if elec_feats["n_anomalies"] > 2:
            recommendations.append("□ Multiple anomaly periods — ask about equipment changes, occupancy")
        if df_temp is not None:
            df_m = merge_temp(df_elec, df_temp)
            if not df_m.empty:
                df_m["daily"] = df_m["consumption"] / df_m["days"]
                r = df_m["daily"].corr(df_m["temp_delta"])
                if abs(r) > 0.6:
                    recommendations.append("□ Strong HVAC dependency — check thermostat settings, duct leaks")
    
    # Check for high water usage
    df_water = loader.get_division("Water")
    if not df_water.empty:
        water_feats = MeterFeatures(df_water).compute()
        if water_feats["daily_avg"] and water_feats["daily_avg"] > 150:
            recommendations.append("□ High water usage — check for leaks, irrigation system")
    
    # Check for gas
    df_gas = loader.get_division("Gas")
    if not df_gas.empty:
        gas_feats = MeterFeatures(df_gas).compute()
        recommendations.append("□ Gas service present — inspect water heater, furnace, appliances")
    
    # AMI-specific
    if ami_file and 'ami_feats' in dir():
        if ami_feats["peak_kw"] > ami_feats["base_kw"] * 10:
            recommendations.append("□ High peak-to-base ratio — look for cycling equipment issues")
    
    # Standard items
    recommendations.extend([
        "□ Verify thermostat type and settings",
        "□ Check air filter condition",
        "□ Inspect windows and doors for air leaks",
        "□ Document appliance ages and conditions",
        "□ Review lighting (LED conversion opportunities)",
    ])
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print_header("END OF REPORT", "═")
    
    if save_charts:
        print(f"\n  Charts saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GRU Energy Audit Pre-Survey Report Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gru_audit.py customer_file.xlsx
  python gru_audit.py customer_file.xlsx --ami ami_data.xlsx
  python gru_audit.py customer_file.xlsx --ami ami_data.xlsx --save

Note: The customer file must have a 'Consumption History' tab with detailed
meter reading data (Division, MR Date, Days, Consumption, etc.)
        """
    )
    
    parser.add_argument(
        "meter_file",
        help="Path to customer meter reading Excel file"
    )
    parser.add_argument(
        "--ami", "-a",
        dest="ami_file",
        help="Path to AMI interval data file (optional)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save charts to disk instead of displaying"
    )
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.meter_file):
        print(f"Error: Meter file not found: {args.meter_file}")
        sys.exit(1)
    
    if args.ami_file and not os.path.exists(args.ami_file):
        print(f"Warning: AMI file not found: {args.ami_file}")
        args.ami_file = None
    
    # Run analysis with error handling
    try:
        generate_presurvey_report(
            meter_file=args.meter_file,
            ami_file=args.ami_file,
            save_charts=args.save
        )
    except ValueError as e:
        print(f"\n  ⚠ Error: {e}")
        print("\n  This tool requires customer files with a populated 'Consumption History' tab.")
        print("  Please ensure the file contains detailed meter reading data.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ⚠ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
