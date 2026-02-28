# ═══════════════════════════════════════════════════════════════════
# GRU Energy Audit Analyzer — Streamlit App
# Converted from Consumption_Project_V1.ipynb
# Includes: Sections 1–7, Auditor Action List, Cross-Utility Correlation
# Excludes: Section 6 (Batch), Section 8 (Customer Mapping)
# ═══════════════════════════════════════════════════════════════════

import io
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRU Energy Audit Analyzer",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
GAINESVILLE_LAT      = 29.6516
GAINESVILLE_LON      = -82.3248
COMFORT_BASE         = 65
MIN_HISTORY_PERIODS  = 8
RESIDUAL_Z_THRESHOLD = 2.5
PERSISTENCE_PERIODS  = 2


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Meter Reading Classes
# ═══════════════════════════════════════════════════════════════════

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
    NON_READ_TYPES   = {"automatic estimation"}

    def __init__(self, fileobj):
        self.fileobj       = fileobj
        self.df            = None
        self.has_mr_reason = False

    def _find_sheet(self, xl):
        for name in xl.sheet_names:
            if "consumption" in name.lower():
                return name
        raise ValueError(f"No consumption sheet found. Sheets: {xl.sheet_names}")

    def _find_header_row(self, xl, sheet):
        for i in range(5):
            df = pd.read_excel(xl, sheet_name=sheet, header=i, nrows=1)
            df.columns = df.columns.str.strip()
            if "Division" in df.columns:
                return i
        return 0

    def load_and_clean(self):
        xl         = pd.ExcelFile(self.fileobj)
        sheet      = self._find_sheet(xl)
        header_row = self._find_header_row(xl, sheet)
        df         = pd.read_excel(xl, sheet_name=sheet, header=header_row)
        df.columns = df.columns.str.strip()
        df         = df.rename(columns=self.COLUMN_MAP)

        self.has_mr_reason = "mr_reason" in df.columns
        df["mr_date"] = pd.to_datetime(df["mr_date"], errors="coerce")

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
            if "mr_type" in df.columns:
                df = df[~df["mr_type"].str.strip().str.lower().isin(self.NON_READ_TYPES)]
            df = df[df["consumption"] > 0]

        df = df[df["days"] > 0]
        df = df.sort_values(["division", "device", "mr_date"]).reset_index(drop=True)
        self.df = df
        return df

    def get_division(self, name):
        if self.df is None:
            raise RuntimeError("Call load_and_clean() first.")
        sub = self.df[self.df["division"] == name].copy()
        if not sub.empty:
            sub = sub[sub["mr_date"] > sub["mr_date"].min()].reset_index(drop=True)
        return sub


class MeterFeatures:
    def __init__(self, df):
        self.df = df.copy().sort_values("mr_date").reset_index(drop=True)

    def compute_features(self):
        df = self.df

        avg_read_interval = df["days"].mean()
        total_consumption = df["consumption"].sum()
        total_days        = df["days"].sum()
        overall_daily_avg = total_consumption / total_days if total_days > 0 else None
        peak_consumption  = df["consumption"].max()
        base_consumption  = df["consumption"].quantile(0.05)
        period_series     = df.set_index("mr_date")["consumption"]
        rolling_avg       = period_series.rolling(window=3).mean()
        daily_avg_series  = df.set_index("mr_date")["avg_daily"] if "avg_daily" in df.columns else None

        iso_cols = [c for c in ["consumption", "days", "avg_daily"] if c in df.columns]
        iso_data = df[iso_cols].dropna()
        df["anomaly"] = False
        if len(iso_data) >= 5:
            preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(iso_data)
            df.loc[iso_data.index, "anomaly"] = (preds == -1)

        n_anomalies   = int(df["anomaly"].sum())
        unit          = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""
        quality_score = self._compute_quality_score(df)

        return {
            "avg_read_interval" : avg_read_interval,
            "total_consumption" : total_consumption,
            "overall_daily_avg" : overall_daily_avg,
            "peak_consumption"  : peak_consumption,
            "base_consumption"  : base_consumption,
            "period_series"     : period_series,
            "rolling_avg"       : rolling_avg,
            "daily_avg_series"  : daily_avg_series,
            "df_with_anomalies" : df,
            "n_anomalies"       : n_anomalies,
            "unit"              : unit,
            "quality_score"     : quality_score,
        }

    def _compute_quality_score(self, df):
        score = 100
        if df["consumption"].isna().any():     score -= 10
        if df["days"].std() > 10:              score -= 5
        if len(df) < 12:                       score -= 15
        if (df["consumption"] == 0).sum() > 2: score -= 10
        if len(df) > 1:
            gaps = df["mr_date"].diff().dt.days
            if gaps.max() > 60:                score -= 20
        return max(0, score)


class MeterGraphs:
    def __init__(self, feats, title_prefix=""):
        self.feats  = feats
        self.prefix = title_prefix
        self.df     = feats["df_with_anomalies"]

    def _get_meter_changes(self):
        if "mr_reason" not in self.df.columns:
            return []
        df      = self.df.sort_values("mr_date")
        reasons = df["mr_reason"].tolist()
        dates   = df["mr_date"].tolist()
        changes = []
        i = 0
        while i < len(reasons):
            if reasons[i] == 22:
                for j in range(i + 1, min(i + 4, len(reasons))):
                    if reasons[j] == 21:
                        changes.append((dates[i], dates[j]))
                        i = j + 1
                        break
                else:
                    changes.append((dates[i], dates[i]))
                    i += 1
            elif reasons[i] == 21:
                found = False
                for j in range(i - 1, max(i - 4, -1), -1):
                    if reasons[j] == 22:
                        found = True
                        break
                if not found:
                    changes.append((dates[i], dates[i]))
                i += 1
            else:
                i += 1
        return changes

    def _add_markers(self, ax):
        df = self.df
        if "mr_reason" not in df.columns:
            return
        move_ins     = df[df["mr_reason"] == 6]
        first_movein = True
        for _, row in move_ins.iterrows():
            lbl = "Move-In" if first_movein else "_nolegend_"
            ax.axvline(x=row["mr_date"], color="dodgerblue",
                       linewidth=1.8, linestyle="--", alpha=0.9, label=lbl)
            first_movein = False

        changes      = self._get_meter_changes()
        first_change = True
        for date_start, date_end in changes:
            lbl = "Meter Change" if first_change else "_nolegend_"
            if date_start == date_end:
                ax.axvline(x=date_start, color="darkorange",
                           linewidth=1.8, linestyle="--", alpha=0.9, label=lbl)
            else:
                ax.axvspan(date_start, date_end, color="darkorange", alpha=0.18, label=lbl)
                ax.axvline(x=date_start, color="darkorange", linewidth=1.2, linestyle="--", alpha=0.6)
                ax.axvline(x=date_end,   color="darkorange", linewidth=1.2, linestyle="--", alpha=0.6)
            first_change = False

    def plot_consumption(self):
        df_plot = self.df[self.df["consumption"] > 0]
        s = df_plot.set_index("mr_date")["consumption"]
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(s.index, s.values, width=20, color="steelblue", alpha=0.85, label="Consumption")
        self._add_markers(ax)
        ax.set_title(f"{self.prefix} — Consumption per Read Period")
        ax.set_ylabel(self.feats["unit"])
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return _fig_to_img(fig)

    def plot_daily_average(self):
        s = self.feats["daily_avg_series"]
        if s is None:
            return None
        s = s[s > 0]
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(s.index, s.values, color="goldenrod", linewidth=2,
                marker="o", markersize=4, label="Daily Avg")
        self._add_markers(ax)
        ax.set_title(f"{self.prefix} — Average Daily Usage per Period")
        ax.set_ylabel(f"{self.feats['unit']}/day")
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return _fig_to_img(fig)

    def plot_rolling_average(self):
        df_plot = self.df[self.df["consumption"] > 0]
        s = df_plot.set_index("mr_date")["consumption"]
        r = s.rolling(window=3).mean()
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(s.index, s.values, color="steelblue", alpha=0.4, linewidth=1.5, label="Consumption")
        ax.plot(r.index, r.values, color="crimson", linewidth=2.5, label="3-Read Rolling Avg")
        self._add_markers(ax)
        ax.set_title(f"{self.prefix} — Consumption Trend")
        ax.set_ylabel(self.feats["unit"])
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return _fig_to_img(fig)

    def plot_anomalies(self):
        df      = self.df[self.df["consumption"] > 0]
        normal  = df[~df["anomaly"]]
        anomaly = df[df["anomaly"]]
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(normal["mr_date"],  normal["consumption"],
               width=20, color="steelblue", alpha=0.85, label="Normal")
        ax.bar(anomaly["mr_date"], anomaly["consumption"],
               width=20, color="crimson",   alpha=0.9,  label="Anomaly (Isolation Forest)")
        self._add_markers(ax)
        ax.set_title(f"{self.prefix} — Anomaly Detection (Isolation Forest)")
        ax.set_ylabel(self.feats["unit"])
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return _fig_to_img(fig), anomaly


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: AMI Classes
# ═══════════════════════════════════════════════════════════════════

class AMILoader:
    SHEET_TYPE_MAP = {
        "ELECTRIC": "Electric", "Electric": "Electric",
        "Sheet1"  : "Electric", "SHEET1"  : "Electric",
        "WATER"   : "Water",    "Water"   : "Water",
        "GAS"     : "Gas",      "Gas"     : "Gas",
    }
    UNIT_MAP = {"Electric": "kWh", "Water": "Gal", "Gas": "CCF"}

    def __init__(self, fileobj):
        self.fileobj   = fileobj
        self.df        = None
        self.util_type = None
        self.unit      = None

    def _find_sheet(self, xl):
        for name in xl.sheet_names:
            if name in self.SHEET_TYPE_MAP:
                return name
        return xl.sheet_names[0]

    def load_and_clean(self):
        xl    = pd.ExcelFile(self.fileobj)
        sheet = self._find_sheet(xl)
        self.util_type = self.SHEET_TYPE_MAP.get(sheet, "Electric")
        self.unit      = self.UNIT_MAP[self.util_type]

        df = pd.read_excel(xl, sheet_name=sheet, header=None, skiprows=4)
        df = df[[0, 1]].copy()
        df.columns = ["timestamp", "raw_value"]

        df["timestamp"] = (df["timestamp"].astype(str)
                           .str.replace(r"\s+EST.*$", "", regex=True)
                           .str.replace(r"\s+EDT.*$", "", regex=True)
                           .str.strip())
        df["timestamp"] = pd.to_datetime(df["timestamp"],
                                         format="%b %d, %Y - %I:%M %p",
                                         errors="coerce")
        df["value_raw"] = (df["raw_value"].astype(str)
                           .str.replace(",", "", regex=False)
                           .str.extract(r"([\d.]+)")[0])
        df["value_raw"] = pd.to_numeric(df["value_raw"], errors="coerce")
        df["value"]     = df["value_raw"] / 1000 if self.util_type == "Electric" else df["value_raw"]
        df["kwh"]       = df["value"]

        df = df.dropna(subset=["timestamp", "kwh"])
        df = df[df["kwh"] > 0]
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        return df


class AMIFeatures:
    def __init__(self, df):
        self.df = df.copy()

    def compute(self):
        df               = self.df.sort_values("timestamp")
        deltas           = df["timestamp"].diff().dropna()
        interval         = deltas.mode()[0]
        interval_minutes = int(interval.total_seconds() / 60)
        base_load        = df["kwh"].quantile(0.05)
        base_load_kw     = base_load / (interval_minutes / 60)
        peak_kwh         = df["kwh"].max()
        peak_kw          = peak_kwh / (interval_minutes / 60)
        df["date"]       = df["timestamp"].dt.date
        daily_series     = df.groupby("date")["kwh"].sum()
        daily_avg_kwh    = daily_series.mean()
        peak_day         = pd.Timestamp(daily_series.idxmax())
        df["hour"]       = df["timestamp"].dt.hour
        avg_by_hour      = df.groupby("hour")["kwh"].mean()

        return {
            "interval_minutes": interval_minutes,
            "base_load"       : base_load,
            "base_load_kw"    : base_load_kw,
            "peak_kwh"        : peak_kwh,
            "peak_kw"         : peak_kw,
            "daily_avg_kwh"   : daily_avg_kwh,
            "daily_series"    : daily_series,
            "peak_day"        : peak_day,
            "avg_by_hour"     : avg_by_hour,
            "df"              : df,
        }


class AMIGraphs:
    def __init__(self, feats, title_prefix="", unit="kWh"):
        self.feats  = feats
        self.prefix = title_prefix
        self.unit   = unit

    def plot_load_shape(self):
        df       = self.feats["df"]
        base_kwh = self.feats["base_load"]
        peak_kwh = self.feats["peak_kwh"]
        base_kw  = self.feats["base_load_kw"]
        peak_kw  = self.feats["peak_kw"]
        fig, ax  = plt.subplots(figsize=(14, 4))
        ax.plot(df["timestamp"], df["kwh"], color="steelblue", linewidth=0.6, alpha=0.85)
        ax.axhline(base_kwh, color="seagreen", linewidth=1.5, linestyle="--",
                   label=f"Base Load ({base_kw:.3f} kW)")
        ax.axhline(peak_kwh, color="crimson", linewidth=1.5, linestyle="--",
                   label=f"Peak Demand ({peak_kw:.3f} kW)")
        ax.set_title(f"{self.prefix} — Load Shape  ({self.feats['interval_minutes']}-min intervals)")
        ax.set_ylabel(f"{self.unit} per interval")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        plt.tight_layout()
        return _fig_to_img(fig)

    def plot_daily_totals(self):
        ds       = self.feats["daily_series"]
        peak_day = self.feats["peak_day"].date()
        avg      = self.feats["daily_avg_kwh"]
        colors   = ["crimson" if d == peak_day else "steelblue" for d in ds.index]
        fig, ax  = plt.subplots(figsize=(14, 4))
        ax.bar(ds.index, ds.values, color=colors, alpha=0.85, width=0.8)
        ax.axhline(avg, color="darkorange", linewidth=2, linestyle="--",
                   label=f"Daily Avg ({avg:.2f} {self.unit})")
        ax.set_title(f"{self.prefix} — Daily Total Usage  (red = peak day: {peak_day})")
        ax.set_ylabel(f"{self.unit} / day")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        plt.tight_layout()
        return _fig_to_img(fig)

    def plot_hourly_profile(self):
        ah      = self.feats["avg_by_hour"]
        base    = self.feats["base_load"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(ah.index, ah.values, color="mediumpurple", alpha=0.85, width=0.7)
        ax.axhline(base, color="seagreen", linewidth=1.5, linestyle="--",
                   label=f"Base Load ({base:.3f} {self.unit}/interval)")
        ax.set_title(f"{self.prefix} — Average Load Shape by Hour of Day")
        ax.set_xlabel("Hour (0 = midnight)")
        ax.set_ylabel(f"Avg {self.unit} per interval")
        ax.set_xticks(range(0, 24))
        ax.legend(fontsize=8)
        plt.tight_layout()
        return _fig_to_img(fig)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Shared Utilities
# ═══════════════════════════════════════════════════════════════════

def get_master_sheet_info(fileobj):
    try:
        ms = pd.read_excel(fileobj, sheet_name="Master Sheet", header=None)

        def safe_get(row, col):
            try:
                val = ms.iloc[row, col]
                return str(val).strip() if pd.notna(val) else None
            except Exception:
                return None

        row_offset = 0
        cell_0_6   = safe_get(0, 6)
        if cell_0_6 and not any(c.isdigit() for c in str(cell_0_6)):
            row_offset = 1

        def get(row, col):
            return safe_get(row + row_offset, col)

        info = {
            "account"        : get(0, 6),
            "customer_name"  : get(1, 6),
            "own_rent"       : get(2, 6),
            "community"      : get(3, 6),
            "address"        : get(4, 6),
            "city_town"      : get(5, 6),
            "gru_rep"        : get(6, 2),
            "survey_date"    : get(7, 2),
            "survey_time"    : get(8, 2),
            "results_sent_to": get(9, 2),
        }

        if info["survey_date"] and "00:00:00" in str(info["survey_date"]):
            try:
                info["survey_date"] = pd.to_datetime(info["survey_date"]).strftime("%m/%d/%Y")
            except Exception:
                pass

        return info
    except Exception:
        return {}


@st.cache_data(show_spinner="Fetching Gainesville temperature data…")
def get_gainesville_temps(start_date, end_date):
    start  = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end    = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"        : GAINESVILLE_LAT,
        "longitude"       : GAINESVILLE_LON,
        "start_date"      : start,
        "end_date"        : end,
        "daily"           : ["temperature_2m_max", "temperature_2m_min"],
        "temperature_unit": "fahrenheit",
        "timezone"        : "America/New_York",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data    = resp.json()["daily"]
        df_temp = pd.DataFrame({
            "date"    : pd.to_datetime(data["time"]),
            "temp_max": data["temperature_2m_max"],
            "temp_min": data["temperature_2m_min"],
        })
        df_temp["temp_avg"] = (df_temp["temp_max"] + df_temp["temp_min"]) / 2
        return df_temp.set_index("date")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch temperature data: {e}")
        return None


def merge_consumption_temp(df_div, df_temp):
    df = df_div.copy().sort_values("mr_date").reset_index(drop=True)
    df = df[df["consumption"] > 0]
    temp_avgs, temp_maxs, temp_mins = [], [], []
    for _, row in df.iterrows():
        end_date     = row["mr_date"]
        start_date   = end_date - pd.Timedelta(days=int(row["days"]))
        mask         = (df_temp.index >= start_date) & (df_temp.index <= end_date)
        period_temps = df_temp[mask]
        if not period_temps.empty:
            temp_avgs.append(period_temps["temp_avg"].mean())
            temp_maxs.append(period_temps["temp_max"].mean())
            temp_mins.append(period_temps["temp_min"].mean())
        else:
            temp_avgs.append(None)
            temp_maxs.append(None)
            temp_mins.append(None)
    df["temp_avg"] = temp_avgs
    df["temp_max"] = temp_maxs
    df["temp_min"] = temp_mins
    return df.dropna(subset=["temp_avg"])


# ─────────────────────────────────────────────────────────────────
# Temperature Charts
# V-shape model for Electricity/Water; linear for Gas
# Side-by-side chart removed (redundant)
# ─────────────────────────────────────────────────────────────────

def plot_temp_overlay(df_merged, title_prefix=""):
    """Consumption bars with temperature line on dual y-axis."""
    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.bar(df_merged["mr_date"], df_merged["consumption"],
            width=20, color="steelblue", alpha=0.6, label="Consumption")
    ax1.set_ylabel(df_merged["mr_unit"].iloc[0] if "mr_unit" in df_merged.columns else "Usage",
                   color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(df_merged["mr_date"], df_merged["temp_avg"],
             color="crimson", linewidth=2.2, marker="o", markersize=4, label="Avg Temp")
    ax2.fill_between(df_merged["mr_date"],
                     df_merged["temp_min"], df_merged["temp_max"],
                     color="crimson", alpha=0.08, label="Temp Range")
    ax2.set_ylabel("Temperature (°F)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(f"{title_prefix} — Consumption vs Temperature (Overlay)")
    fig.autofmt_xdate()
    plt.tight_layout()
    return _fig_to_img(fig)


def plot_temp_scatter(df_merged, title_prefix="", division="Electricity"):
    """
    V-shape scatter for Electricity/Water: usage vs |temp - 65°F|.
    Captures both summer cooling and winter heating demand in a single r value.

    Linear scatter for Gas: usage vs raw temp (heating load goes up as temp drops).
    The old linear Pearson r is also shown for reference but noted as misleading for HVAC.
    """
    unit = df_merged["mr_unit"].iloc[0] if "mr_unit" in df_merged.columns else ""
    df   = df_merged.copy()
    df["temp_delta"] = (df["temp_avg"] - COMFORT_BASE).abs()

    if "days" in df.columns:
        df["daily_cons"] = df["consumption"] / df["days"]
    else:
        df["daily_cons"] = df["consumption"]

    if division == "Gas":
        r_primary = df["daily_cons"].corr(df["temp_avg"])
        r_vshape  = df["daily_cons"].corr(df["temp_delta"])
        x_col     = "temp_avg"
        xlabel    = "Avg Temperature (°F)  —  expect negative correlation for heating"
        title_r   = f"Linear r = {r_primary:.2f}"
        footnote  = f"V-shape r = {r_vshape:.2f}  (shown for reference)"
        if r_primary < -0.6:
            interp = "Strong heating load — gas usage rises significantly as temperature drops."
        elif r_primary < -0.3:
            interp = "Moderate heating relationship — temperature influences gas usage."
        else:
            interp = "Weak heating relationship — gas may serve water heating or cooking too."
    else:
        # V-shape: captures U-shaped HVAC demand (high in summer AND winter)
        r_primary = df["daily_cons"].corr(df["temp_delta"])
        r_linear  = df["daily_cons"].corr(df["temp_avg"])
        x_col     = "temp_delta"
        xlabel    = "|Temperature − 65°F|  (V-shape: distance from comfort baseline)"
        title_r   = f"V-shape r = {r_primary:.2f}"
        footnote  = (
            f"Linear Pearson r = {r_linear:.2f}  —  "
            "Low linear r is expected and normal for HVAC customers. "
            "Summer cooling and winter heating cancel each other out in the linear model. "
            "The V-shape r captures both humps and is the correct metric here."
        )
        if r_primary > 0.7:
            interp = "Very strong HVAC relationship — usage rises clearly with both heat and cold."
        elif r_primary > 0.5:
            interp = "Moderate HVAC relationship — temperature clearly influences usage."
        elif r_primary > 0.3:
            interp = "Weak HVAC relationship — some temperature sensitivity detected."
        else:
            interp = "Minimal HVAC relationship — usage appears mostly independent of temperature."

    def season_color(temp):
        if temp >= 80: return "#f76f6f"
        if temp <= 55: return "#4f8ef7"
        return "#3ecf8e"

    colors = [season_color(t) for t in df["temp_avg"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df[x_col], df["daily_cons"],
               c=colors, alpha=0.85, edgecolors="white", s=75, zorder=3)
    z      = np.polyfit(df[x_col], df["daily_cons"], 1)
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color="darkorange",
            linewidth=2, linestyle="--", label="Trend")
    ax.legend(handles=[
        mpatches.Patch(color="#f76f6f", label="Hot (>80°F)"),
        mpatches.Patch(color="#4f8ef7", label="Cold (<55°F)"),
        mpatches.Patch(color="#3ecf8e", label="Mild (55–80°F)"),
        mpatches.Patch(color="darkorange", label="Trend line"),
    ], fontsize=8)
    ax.set_title(f"{title_prefix} — Usage vs Temperature  ({title_r})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{unit}/day")
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7.5, color="gray", style="italic",
             wrap=True)
    plt.tight_layout()
    return _fig_to_img(fig), r_primary, interp


# ─────────────────────────────────────────────────────────────────
# Cross-Utility Correlation
# ─────────────────────────────────────────────────────────────────

def plot_cross_utility_correlation(df_elec, df_water, df_gas, name=""):
    """
    Normalized seasonal overlay for all available divisions + pairwise Pearson r.
    Useful for spotting whether electricity/gas move together (resistance heating)
    or inversely (gas heating + electric cooling).
    """
    series     = {}
    color_map  = {
        "Electricity": "steelblue",
        "Water"      : "seagreen",
        "Gas"        : "darkorange",
    }

    def to_daily_series(df_div):
        df = df_div[df_div["consumption"] > 0].copy().sort_values("mr_date")
        if "avg_daily" in df.columns:
            s = df.set_index("mr_date")["avg_daily"].dropna()
        else:
            s = (df["consumption"] / df["days"])
            s.index = df["mr_date"]
        return s

    if not df_elec.empty:  series["Electricity"] = to_daily_series(df_elec)
    if not df_water.empty: series["Water"]        = to_daily_series(df_water)
    if not df_gas.empty:   series["Gas"]           = to_daily_series(df_gas)

    if len(series) < 2:
        return None, {}

    # Normalized overlay
    fig, ax = plt.subplots(figsize=(13, 4))
    for div, s in series.items():
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        ax.plot(s.index, s_norm, color=color_map[div], linewidth=2,
                marker="o", markersize=3, alpha=0.85, label=div)
    ax.set_title(f"{name} — Cross-Utility Seasonal Pattern (Normalized 0–1)")
    ax.set_ylabel("Normalized Daily Usage (0 = min, 1 = peak)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    overlay_img = _fig_to_img(fig)

    # Pairwise Pearson r on monthly averages
    monthly  = {}
    for div, s in series.items():
        s2       = s.copy()
        s2.index = pd.to_datetime(s2.index)
        monthly[div] = s2.resample("MS").mean()

    pairs    = {}
    div_list = list(monthly.keys())
    for i in range(len(div_list)):
        for j in range(i + 1, len(div_list)):
            a, b   = div_list[i], div_list[j]
            merged = pd.concat([monthly[a], monthly[b]], axis=1, join="inner").dropna()
            if len(merged) >= 4:
                r = merged.iloc[:, 0].corr(merged.iloc[:, 1])
                pairs[f"{a} ↔ {b}"] = round(r, 3)

    return overlay_img, pairs


# ─────────────────────────────────────────────────────────────────
# Year-over-Year
# ─────────────────────────────────────────────────────────────────

def compute_year_over_year(df_div):
    df = df_div[df_div["consumption"] > 0].copy().sort_values("mr_date")
    if df.empty or len(df) < 2:
        return None
    latest  = df["mr_date"].max()
    cutoff  = latest - pd.DateOffset(years=1)
    cutoff2 = cutoff  - pd.DateOffset(years=1)
    recent  = df[df["mr_date"] > cutoff]
    prior   = df[(df["mr_date"] > cutoff2) & (df["mr_date"] <= cutoff)]
    if recent.empty or prior.empty:
        return None

    def daily_avg(sub):
        tc = sub["consumption"].sum()
        td = sub["days"].sum()
        return tc / td if td > 0 else None

    recent_avg = daily_avg(recent)
    prior_avg  = daily_avg(prior)
    if recent_avg is None or prior_avg is None or prior_avg == 0:
        return None

    pct_change = (recent_avg - prior_avg) / prior_avg * 100
    direction  = "UP ▲" if pct_change > 2 else "DOWN ▼" if pct_change < -2 else "STABLE ↔"
    unit       = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""

    def safe_daily(sub):
        if "avg_daily" in sub.columns:
            return sub["avg_daily"].clip(lower=0)
        return sub["consumption"] / sub["days"]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(recent["mr_date"], safe_daily(recent),
           width=20, color="steelblue", alpha=0.85, label="Recent 12 Months")
    ax.bar(prior["mr_date"],  safe_daily(prior),
           width=20, color="goldenrod",  alpha=0.7,  label="Prior 12 Months")
    ax.set_title("Year-over-Year Daily Average Usage")
    ax.set_ylabel(f"{unit}/day")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    return {
        "recent_avg": recent_avg,
        "prior_avg" : prior_avg,
        "pct_change": pct_change,
        "direction" : direction,
        "unit"      : unit,
        "chart"     : _fig_to_img(fig),
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: Weather-Normalized Anomaly Framework
# ═══════════════════════════════════════════════════════════════════

def compute_degree_days(temp_avg, base=COMFORT_BASE):
    hdd = max(0, base - temp_avg)
    cdd = max(0, temp_avg - base)
    return hdd, cdd


def build_single_customer_anomaly(df_div, df_temp, division="Electricity"):
    """
    Single-customer weather-normalized anomaly detection using rolling
    HuberRegressor on Heating/Cooling Degree Day features.

    Returns a DataFrame with residual z-scores, CI bands, and anomaly flags.
    Requires at least MIN_HISTORY_PERIODS periods after temperature matching.
    """
    df = df_div[df_div["consumption"] > 0].copy().sort_values("mr_date").reset_index(drop=True)

    # Build period-level HDD/CDD features
    period_data = []
    for _, row in df.iterrows():
        end_date   = row["mr_date"]
        start_date = end_date - pd.Timedelta(days=int(row["days"]))
        mask       = (df_temp.index >= start_date) & (df_temp.index <= end_date)
        temp_slice = df_temp.loc[mask]
        if temp_slice.empty:
            continue
        temp_avg    = temp_slice["temp_avg"].mean()
        hdd, cdd    = compute_degree_days(temp_avg)
        period_data.append({
            "mr_date"  : end_date,
            "kwh"      : row["consumption"],
            "days"     : row["days"],
            "daily_kwh": row["consumption"] / row["days"],
            "temp_avg" : temp_avg,
            "hdd"      : hdd,
            "cdd"      : cdd,
        })

    df_acc = pd.DataFrame(period_data)
    if len(df_acc) < MIN_HISTORY_PERIODS:
        return pd.DataFrame()

    df_acc = df_acc.sort_values("mr_date").reset_index(drop=True)
    rows   = []

    for i in range(MIN_HISTORY_PERIODS, len(df_acc)):
        hist    = df_acc.iloc[:i]
        current = df_acc.iloc[i]
        X_hist  = hist[["hdd", "cdd"]].values
        y_hist  = hist["daily_kwh"].values
        X_curr  = np.array([[current["hdd"], current["cdd"]]])

        model = HuberRegressor(epsilon=1.35)
        model.fit(X_hist, y_hist)

        predicted      = model.predict(X_curr)[0]
        residual       = current["daily_kwh"] - predicted
        hist_residuals = y_hist - model.predict(X_hist)
        resid_std      = np.std(hist_residuals) if np.std(hist_residuals) > 0 else 1
        resid_z        = residual / resid_std
        n              = len(hist)
        se_pred        = resid_std * np.sqrt(1 + 1 / n)

        rows.append({
            "mr_date"        : current["mr_date"],
            "actual_daily"   : current["daily_kwh"],
            "predicted_daily": predicted,
            "residual"       : residual,
            "residual_z"     : resid_z,
            "ci_lower"       : predicted - 1.96 * se_pred,
            "ci_upper"       : predicted + 1.96 * se_pred,
            "temp_avg"       : current["temp_avg"],
            "hdd"            : current["hdd"],
            "cdd"            : current["cdd"],
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    df_out["anomaly_high"] = df_out["residual_z"] >  RESIDUAL_Z_THRESHOLD
    df_out["anomaly_low"]  = df_out["residual_z"] < -RESIDUAL_Z_THRESHOLD
    df_out["anomaly"]      = df_out["anomaly_high"] | df_out["anomaly_low"]

    df_out["persistent_high"] = (
        df_out["anomaly_high"].rolling(PERSISTENCE_PERIODS).sum() >= PERSISTENCE_PERIODS
    )
    df_out["persistent_low"] = (
        df_out["anomaly_low"].rolling(PERSISTENCE_PERIODS).sum() >= PERSISTENCE_PERIODS
    )
    df_out["persistent"] = df_out["persistent_high"] | df_out["persistent_low"]

    return df_out


def plot_weather_anomaly(df_anomaly, title_prefix="", unit="kWh"):
    """
    Two-panel chart:
      Top    — Actual vs Predicted daily usage with 95% CI band
      Bottom — Residual Z-score bars colored by status
    """
    if df_anomaly.empty:
        return None

    df     = df_anomaly.sort_values("mr_date")
    normal = df[~df["anomaly"]]
    high   = df[df["anomaly_high"]]
    low    = df[df["anomaly_low"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Top: actual vs predicted
    ax1.fill_between(df["mr_date"], df["ci_lower"], df["ci_upper"],
                     alpha=0.2, color="steelblue", label="95% CI")
    ax1.plot(df["mr_date"], df["predicted_daily"], color="steelblue",
             linewidth=2, linestyle="--", label="Weather-Predicted")
    ax1.scatter(normal["mr_date"], normal["actual_daily"],
                color="gray", s=50, zorder=5, label="Normal")
    ax1.scatter(high["mr_date"], high["actual_daily"],
                color="crimson", s=90, zorder=6, label="High Anomaly", marker="^")
    ax1.scatter(low["mr_date"],  low["actual_daily"],
                color="dodgerblue", s=90, zorder=6, label="Low Anomaly", marker="v")
    ax1.set_title(f"{title_prefix} — Weather-Normalized Anomaly Detection (HDD/CDD Regression)")
    ax1.set_ylabel(f"{unit}/day")
    ax1.legend(fontsize=8, loc="upper left")

    # Bottom: residual z-scores
    bar_colors = [
        "crimson"    if z >  RESIDUAL_Z_THRESHOLD else
        "dodgerblue" if z < -RESIDUAL_Z_THRESHOLD else
        "gray"
        for z in df["residual_z"]
    ]
    ax2.bar(df["mr_date"], df["residual_z"], width=20, color=bar_colors, alpha=0.75)
    ax2.axhline( RESIDUAL_Z_THRESHOLD, color="crimson",    linestyle="--", linewidth=1.5,
                 label=f"+{RESIDUAL_Z_THRESHOLD}σ")
    ax2.axhline(-RESIDUAL_Z_THRESHOLD, color="dodgerblue", linestyle="--", linewidth=1.5,
                 label=f"-{RESIDUAL_Z_THRESHOLD}σ")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Residual Z-Score  (weather-adjusted deviation from predicted)")
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8)

    fig.autofmt_xdate()
    plt.tight_layout()
    return _fig_to_img(fig)


# ─────────────────────────────────────────────────────────────────
# Section 7.4 — Auditor Action List
# ─────────────────────────────────────────────────────────────────

def generate_auditor_action_list(df_anomaly, customer_name="", account="", unit="kWh"):
    """
    Generate a prioritized auditor action list from single-customer
    weather-normalized anomaly results.

    Priority levels:
      🔴 HIGH   — Persistent anomaly |z| > 4
      🟠 MEDIUM — Persistent anomaly |z| 2.5–4  OR  5+ historical high periods
      🟡 REVIEW — Single-period anomaly |z| > 3
      ✅ NORMAL — No flags detected
    """
    if df_anomaly.empty:
        return []

    latest  = df_anomaly.sort_values("mr_date").iloc[-1]
    n_high  = int(df_anomaly["anomaly_high"].sum())
    n_low   = int(df_anomaly["anomaly_low"].sum())

    z         = latest["residual_z"]
    actual    = latest["actual_daily"]
    predicted = latest["predicted_daily"]
    last_read = latest["mr_date"].strftime("%Y-%m-%d")

    actions = []

    # ── 🔴 HIGH ──────────────────────────────────────────────────
    if latest["persistent_high"] and z > 4:
        actions.append({
            "priority"         : "🔴 HIGH",
            "issue"            : "Extreme Persistent High Usage",
            "detail"           : f"Using {actual:.1f} {unit}/day vs {predicted:.1f} predicted",
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "check HVAC, water heater, insulation",
            "potential_savings": f"{(actual - predicted) * 30:.0f} {unit}/month",
        })
    elif latest["persistent_low"] and z < -4:
        actions.append({
            "priority"         : "🔴 HIGH",
            "issue"            : "Extreme Persistent Low Usage",
            "detail"           : f"Using {actual:.1f} {unit}/day vs {predicted:.1f} predicted",
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "Verify meter function — check for vacancy or solar install",
            "potential_savings": "N/A — verify meter",
        })

    # ── 🟠 MEDIUM ────────────────────────────────────────────────
    elif latest["persistent_high"] and 2.5 < z <= 4:
        actions.append({
            "priority"         : "🟠 MEDIUM",
            "issue"            : "Elevated Usage Pattern",
            "detail"           : f"Using {actual:.1f} {unit}/day vs {predicted:.1f} predicted",
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "discuss usage patterns, HVAC maintenance",
            "potential_savings": f"{(actual - predicted) * 30:.0f} {unit}/month",
        })
    elif latest["persistent_low"] and -4 <= z < -2.5:
        actions.append({
            "priority"         : "🟠 MEDIUM",
            "issue"            : "Persistent Low Usage",
            "detail"           : f"Using {actual:.1f} {unit}/day vs {predicted:.1f} predicted",
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "Verify occupancy status — confirm meter accuracy",
            "potential_savings": "N/A",
        })

    # Recurring historical pattern (not currently persistent)
    if n_high >= 5 and not latest["persistent_high"]:
        actions.append({
            "priority"         : "🟠 MEDIUM",
            "issue"            : "Recurring High Usage Events",
            "detail"           : f"{n_high} high anomaly periods in history",
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "Review full billing history — identify seasonal equipment issues",
            "potential_savings": "Varies",
        })

    # ── 🟡 REVIEW ────────────────────────────────────────────────
    if latest["anomaly_high"] and not latest["persistent_high"] and z > 3:
        actions.append({
            "priority"         : "🟡 REVIEW",
            "issue"            : "Recent Single-Period High Usage",
            "detail"           : f"Spike: {actual:.1f} {unit}/day  (z = {z:.2f})",
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "may self-resolve",
            "potential_savings": "TBD",
        })

    # ── ✅ NORMAL ─────────────────────────────────────────────────
    if not actions:
        actions.append({
            "priority"         : "✅ NORMAL",
            "issue"            : "No anomalies detected",
            "detail"           : (
                f"Latest z-score: {z:.2f}  |  "
                f"{n_high} historical high periods, {n_low} historical low periods"
            ),
            "z_score"          : round(z, 2),
            "last_read"        : last_read,
            "action"           : "usage consistent with weather patterns",
            "potential_savings": "—",
        })

    return actions


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

st.title("Energy Audit Analyzer")
st.caption("Energy & Water Savings Plan  |  Internal Analysis Tool  |  2026")

tab_meter, tab_ami = st.tabs(["📊 Meter Reading Analysis", "🔬 AMI Interval Analysis"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — METER READING ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_meter:
    st.header("Meter Reading Analysis")
    uploaded = st.file_uploader(
        "Upload a GRU Customer Excel file (.xlsx)",
        type=["xlsx"],
        key="meter_upload",
        help=(
            "File must contain a Consumption (or Consumption History) sheet. "
            "Optionally include a Master Sheet for customer info."
        ),
    )

    if uploaded:
        with st.spinner("Loading and processing file…"):
            file_bytes = uploaded.read()
            info       = get_master_sheet_info(io.BytesIO(file_bytes))
            loader     = MeterLoader(io.BytesIO(file_bytes))
            try:
                loader.load_and_clean()
            except Exception as e:
                st.error(f"❌ Could not load meter data: {e}")
                st.stop()

            df_elec  = loader.get_division("Electricity")
            df_water = loader.get_division("Water")
            df_gas   = loader.get_division("Gas")

        # ── Customer Info Card ────────────────────────────────────
        if info:
            st.subheader("👤 Customer Information")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customer",    info.get("customer_name")   or "—")
            c2.metric("Account",     info.get("account")         or "—")
            c3.metric("Address",     info.get("address")         or "—")
            c4.metric("GRU Rep",     info.get("gru_rep")         or "—")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Own / Rent",  info.get("own_rent")        or "—")
            c6.metric("Community",   info.get("community")       or "—")
            c7.metric("Survey Date", info.get("survey_date")     or "—")
            c8.metric("Results To",  info.get("results_sent_to") or "—")
            st.divider()

        # ── Division Selector ─────────────────────────────────────
        available_divs = []
        if not df_elec.empty:  available_divs.append("Electricity")
        if not df_water.empty: available_divs.append("Water")
        if not df_gas.empty:   available_divs.append("Gas")

        if not available_divs:
            st.warning("No meter reading data found in this file.")
            st.stop()

        division = st.selectbox("Select Division to Analyze", available_divs)
        df_div   = {"Electricity": df_elec, "Water": df_water, "Gas": df_gas}[division]
        name     = info.get("customer_name", "") if info else ""
        prefix   = f"{name} — {division}" if name else division

        with st.spinner("Computing features…"):
            feats  = MeterFeatures(df_div).compute_features()
            graphs = MeterGraphs(feats, title_prefix=prefix)
        unit = feats["unit"]

        # ── Summary Metrics ───────────────────────────────────────
        st.subheader(f"📈 {division} Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Consumption",
                  f"{feats['total_consumption']:,.1f} {unit}")
        m2.metric("Daily Average",
                  f"{feats['overall_daily_avg']:.2f} {unit}/day" if feats["overall_daily_avg"] else "N/A")
        m3.metric("Peak Period",
                  f"{feats['peak_consumption']:,.1f} {unit}")
        m4.metric("Anomalies (IF)",      feats["n_anomalies"])
        m5.metric("Data Quality Score",  f"{feats['quality_score']}/100")
        st.divider()

        # ── Consumption Charts ────────────────────────────────────
        st.subheader("📊 Consumption Charts")
        st.image(graphs.plot_consumption(), use_container_width=True)
        img_daily = graphs.plot_daily_average()
        if img_daily:
            st.image(img_daily, use_container_width=True)
        st.image(graphs.plot_rolling_average(), use_container_width=True)
        st.divider()

        # ── Isolation Forest Anomaly Detection ───────────────────
        st.subheader("🚨 Anomaly Detection — Isolation Forest")
        st.caption(
            "Isolation Forest flags periods that are statistically unusual based on "
            "consumption, days, and daily average — without accounting for weather. "
            "The weather-normalized model in Section 7 (below) is more precise."
        )
        anomaly_img, anomaly_df = graphs.plot_anomalies()
        st.image(anomaly_img, use_container_width=True)
        if not anomaly_df.empty:
            cols = [c for c in ["mr_date", "mr_reason", "mr_type", "days",
                                "consumption", "avg_daily"] if c in anomaly_df.columns]
            st.dataframe(anomaly_df[cols].reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No anomalous periods flagged by Isolation Forest.")
        st.divider()

        # ── Year-over-Year ────────────────────────────────────────
        st.subheader("📅 Year-over-Year Comparison")
        yoy = compute_year_over_year(df_div)
        if yoy:
            y1, y2, y3 = st.columns(3)
            y1.metric("Recent 12-Mo Daily Avg", f"{yoy['recent_avg']:.2f} {yoy['unit']}/day")
            y2.metric("Prior 12-Mo Daily Avg",  f"{yoy['prior_avg']:.2f} {yoy['unit']}/day")
            y3.metric("YoY Change", f"{yoy['pct_change']:+.1f}%  {yoy['direction']}")
            st.image(yoy["chart"], use_container_width=True)
        else:
            st.info("Insufficient data for year-over-year comparison (requires 2+ years).")
        st.divider()

        # ── Temperature Correlation ───────────────────────────────
        st.subheader("🌡️ Temperature Correlation — Gainesville FL")
        st.caption(
            "**Electricity & Water** use a V-shape model: usage is correlated against "
            "|temp − 65°F| (distance from comfort baseline), which captures both "
            "summer cooling demand and winter heating demand in a single r value. "
            "A low linear Pearson r is expected and normal for HVAC customers — "
            "the two seasonal peaks cancel each other out in a linear model. "
            "**Gas** uses a linear model (heating load rises as temperature drops)."
        )

        with st.spinner("Fetching weather data…"):
            df_temp = get_gainesville_temps(
                str(df_div["mr_date"].min().date()),
                str(df_div["mr_date"].max().date()),
            )

        if df_temp is not None:
            df_merged = merge_consumption_temp(df_div, df_temp)
            if not df_merged.empty:
                st.image(plot_temp_overlay(df_merged, title_prefix=prefix),
                         use_container_width=True)
                scatter_img, r_val, interp = plot_temp_scatter(
                    df_merged, title_prefix=prefix, division=division)
                st.image(scatter_img, use_container_width=True)
                st.info(f"**Primary correlation: {r_val:.2f}** — {interp}")
            else:
                st.warning("No overlapping temperature/consumption data found.")
        else:
            st.warning("Could not fetch temperature data. Check internet connection.")
        st.divider()

        # ── Cross-Utility Correlation ─────────────────────────────
        st.subheader("🔗 Cross-Utility Seasonal Correlation")
        st.caption(
            "Normalized daily usage for all available divisions on one axis. "
            "Useful for spotting whether electricity and gas move together "
            "(resistance or dual-fuel heating) or inversely (gas winter / electric summer). "
            "Pairwise Pearson r computed on monthly averages."
        )
        overlay_img, pairs = plot_cross_utility_correlation(
            df_elec, df_water, df_gas, name=name)
        if overlay_img:
            st.image(overlay_img, use_container_width=True)
            if pairs:
                pair_cols = st.columns(len(pairs))
                for col_r, (label, r) in zip(pair_cols, pairs.items()):
                    col_r.metric(label, f"r = {r:.3f}")
                st.caption(
                    "r > 0.7 = strong co-movement  |  "
                    "r near 0 = independent seasonal patterns  |  "
                    "r < −0.4 = inverse seasonal pattern (e.g. gas peaks in winter, electric in summer)"
                )
        else:
            st.info("Cross-utility correlation requires at least two divisions with data in this file.")
        st.divider()

        # ── Section 7: Weather-Normalized Anomaly Framework ──────
        st.subheader("🧠 Weather-Normalized Anomaly Detection (Section 7)")
        st.caption(
            "Trains a rolling **HuberRegressor** (robust to outliers) on Heating Degree Days "
            "and Cooling Degree Days to predict expected usage for each billing period. "
            f"Periods where actual usage deviates more than **±{RESIDUAL_Z_THRESHOLD}σ** "
            "from the weather-adjusted prediction are flagged. "
            f"**Persistent anomalies** ({PERSISTENCE_PERIODS}+ consecutive flagged periods) "
            "are escalated in the Auditor Action List below."
        )

        if df_temp is not None:
            with st.spinner("Running weather-normalized anomaly model…"):
                df_anomaly_wn = build_single_customer_anomaly(df_div, df_temp, division=division)

            if not df_anomaly_wn.empty:
                n_high_wn = int(df_anomaly_wn["anomaly_high"].sum())
                n_low_wn  = int(df_anomaly_wn["anomaly_low"].sum())
                n_pers_wn = int(df_anomaly_wn["persistent"].sum())

                wn1, wn2, wn3, wn4 = st.columns(4)
                wn1.metric("Periods Analyzed", len(df_anomaly_wn))
                wn2.metric("High Anomalies",   n_high_wn)
                wn3.metric("Low Anomalies",    n_low_wn)
                wn4.metric("Persistent",       n_pers_wn)

                wn_img = plot_weather_anomaly(df_anomaly_wn, title_prefix=prefix, unit=unit)
                if wn_img:
                    st.image(wn_img, use_container_width=True)

                if n_high_wn > 0 or n_low_wn > 0:
                    flagged = df_anomaly_wn[df_anomaly_wn["anomaly"]].copy()
                    flagged["type"] = flagged.apply(
                        lambda r: "🔴 High" if r["anomaly_high"] else "🔵 Low", axis=1)
                    flagged["persistent?"] = flagged["persistent"].map({True: "Yes", False: "No"})
                    display_cols = [
                        "mr_date", "type", "actual_daily", "predicted_daily",
                        "residual_z", "temp_avg", "hdd", "cdd", "persistent?",
                    ]
                    st.dataframe(
                        flagged[display_cols].round(2).reset_index(drop=True),
                        use_container_width=True,
                    )
                else:
                    st.success("✅ No weather-normalized anomalies detected.")

                st.divider()

                # ── Auditor Action List ───────────────────────────
                st.subheader("📋 Auditor Action List")
                st.caption(
                    "**🔴 HIGH** = Persistent extreme anomaly — immediate site visit recommended  |  "
                    "**🟠 MEDIUM** = Persistent elevated or recurring pattern — schedule follow-up  |  "
                    "**🟡 REVIEW** = Single-period spike — monitor next billing period  |  "
                    "**✅ NORMAL** = Usage consistent with weather patterns"
                )

                actions = generate_auditor_action_list(
                    df_anomaly_wn,
                    customer_name=name,
                    account=info.get("account", "") if info else "",
                    unit=unit,
                )

                for act in actions:
                    priority = act["priority"]
                    with st.expander(
                        f"{priority}  |  {act['issue']}  "
                        f"|  z = {act['z_score']}  |  Last read: {act['last_read']}"
                    ):
                        col_a, col_b = st.columns(2)
                        col_a.markdown(f"**Detail:** {act['detail']}")
                        col_a.markdown(f"**Recommended Action:** {act['action']}")
                        col_b.markdown(f"**Priority:** {priority}")
                        col_b.markdown(f"**Potential Savings:** {act['potential_savings']}")

            else:
                st.info(
                    f"Not enough temperature-matched data for weather-normalized analysis "
                    f"(minimum {MIN_HISTORY_PERIODS} billing periods required). "
                    f"The file may have too few reads or the temperature API may be unavailable "
                    f"for the date range."
                )
        else:
            st.warning("Weather-normalized analysis requires temperature data — check internet connection.")

        # ── Raw Data Viewer ───────────────────────────────────────
        st.divider()
        with st.expander("🗂️ View Raw Meter Data"):
            st.dataframe(df_div, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — AMI INTERVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_ami:
    st.header("AMI 15-Minute Interval Analysis")
    ami_uploaded = st.file_uploader(
        "Upload a GRU AMI Excel file (.xlsx)",
        type=["xlsx"],
        key="ami_upload",
        help=(
            "Supports Electric, Water, and Gas AMI files. "
            "Sheet name auto-detected (ELECTRIC, Electric, Sheet1, WATER, GAS, etc.). "
            "Timestamp formats with and without EST/EDT suffix are handled."
        ),
    )

    if ami_uploaded:
        with st.spinner("Loading AMI data…"):
            ami_bytes  = ami_uploaded.read()
            ami_loader = AMILoader(io.BytesIO(ami_bytes))
            try:
                df_ami = ami_loader.load_and_clean()
            except Exception as e:
                st.error(f"❌ Could not load AMI data: {e}")
                st.stop()

            unit       = ami_loader.unit
            util_type  = ami_loader.util_type
            ami_feats  = AMIFeatures(df_ami).compute()
            ami_graphs = AMIGraphs(ami_feats, title_prefix=util_type, unit=unit)

        # ── AMI Summary ───────────────────────────────────────────
        st.subheader(f"⚡ AMI Summary — {util_type}")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Interval",    f"{ami_feats['interval_minutes']} min")
        a2.metric("Base Load",   f"{ami_feats['base_load_kw']:.3f} kW")
        a3.metric("Peak Demand", f"{ami_feats['peak_kw']:.3f} kW")
        a4.metric("Daily Avg",   f"{ami_feats['daily_avg_kwh']:.2f} {unit}/day")
        a5.metric("Peak Day",    str(ami_feats["peak_day"].date()))
        a6, a7 = st.columns(2)
        a6.metric("Date Range Start", str(df_ami["timestamp"].min().date()))
        a7.metric("Date Range End",   str(df_ami["timestamp"].max().date()))
        st.divider()

        # ── AMI Charts ────────────────────────────────────────────
        st.subheader("📊 Load Shape (Full Timeline)")
        st.image(ami_graphs.plot_load_shape(), use_container_width=True)

        st.subheader("📊 Daily Totals")
        st.image(ami_graphs.plot_daily_totals(), use_container_width=True)

        st.subheader("📊 Hourly Load Profile")
        st.image(ami_graphs.plot_hourly_profile(), use_container_width=True)
        st.divider()

        # ── AMI Temperature Correlation ───────────────────────────
        st.subheader("🌡️ AMI Temperature Correlation — Gainesville FL")
        st.caption(
            "V-shape model (|temp − 65°F|) for Electricity/Water. "
            "Linear model for Gas. Same methodology as meter reading analysis."
        )

        ami_daily         = ami_feats["daily_series"].reset_index()
        ami_daily.columns = ["date", "kwh"]
        ami_daily["date"] = pd.to_datetime(ami_daily["date"])

        with st.spinner("Fetching weather data…"):
            df_temp_ami = get_gainesville_temps(
                str(ami_daily["date"].min().date()),
                str(ami_daily["date"].max().date()),
            )

        if df_temp_ami is not None:
            df_ami_temp = ami_daily.merge(
                df_temp_ami.reset_index(),
                on="date", how="inner",
            ).dropna(subset=["temp_avg"])

            if not df_ami_temp.empty:
                if util_type == "Gas":
                    df_ami_temp["temp_delta"] = df_ami_temp["temp_avg"]
                    xlabel  = "Avg Temperature (°F) — expect negative for heating load"
                    r_label = "Linear r"
                    r = df_ami_temp["kwh"].corr(df_ami_temp["temp_delta"])
                    interp = (
                        "Strong heating load — usage rises as temperature drops." if r < -0.6
                        else "Moderate heating relationship." if r < -0.3
                        else "Weak heating relationship — gas may serve multiple end uses."
                    )
                else:
                    df_ami_temp["temp_delta"] = (df_ami_temp["temp_avg"] - COMFORT_BASE).abs()
                    xlabel  = "|Temperature − 65°F|  (V-shape: distance from comfort baseline)"
                    r_label = "V-shape r"
                    r = df_ami_temp["kwh"].corr(df_ami_temp["temp_delta"])
                    interp = (
                        "Very strong HVAC relationship." if r > 0.7
                        else "Moderate HVAC relationship." if r > 0.5
                        else "Weak HVAC relationship." if r > 0.3
                        else "Minimal temperature sensitivity."
                    )

                def season_color(temp):
                    if temp >= 80: return "#f76f6f"
                    if temp <= 55: return "#4f8ef7"
                    return "#3ecf8e"

                colors = [season_color(t) for t in df_ami_temp["temp_avg"]]
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(df_ami_temp["temp_delta"], df_ami_temp["kwh"],
                           c=colors, alpha=0.85, edgecolors="white", s=70, zorder=3)
                z_fit  = np.polyfit(df_ami_temp["temp_delta"], df_ami_temp["kwh"], 1)
                x_line = np.linspace(df_ami_temp["temp_delta"].min(),
                                     df_ami_temp["temp_delta"].max(), 100)
                ax.plot(x_line, np.poly1d(z_fit)(x_line), color="darkorange",
                        linewidth=2, linestyle="--", label="Trend")
                ax.legend(handles=[
                    mpatches.Patch(color="#f76f6f", label="Hot (>80°F)"),
                    mpatches.Patch(color="#4f8ef7", label="Cold (<55°F)"),
                    mpatches.Patch(color="#3ecf8e", label="Mild (55–80°F)"),
                    mpatches.Patch(color="darkorange", label="Trend"),
                ], fontsize=8)
                ax.set_title(f"{util_type} AMI Daily Usage vs Temperature  ({r_label} = {r:.2f})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(f"{unit}/day")
                plt.tight_layout()
                st.image(_fig_to_img(fig), use_container_width=True)
                st.info(f"**{r_label} = {r:.2f}** — {interp}")
            else:
                st.warning("No overlapping temperature/AMI data found.")
        else:
            st.warning("Could not fetch temperature data. Check internet connection.")

        # ── Raw AMI Data ──────────────────────────────────────────
        st.divider()
        with st.expander("🗂️ View Raw AMI Data (first 500 rows)"):
            st.dataframe(df_ami.head(500), use_container_width=True)
