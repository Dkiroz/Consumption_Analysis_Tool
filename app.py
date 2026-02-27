"""
Energy Audit Analyzer — Streamlit Web App
All features:
  - Single File Analysis (meter reading) with MR reason markers + temperature charts
  - Year-over-Year comparison
  - AMI Analysis with load shape, daily totals, hourly profile + temperature charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import requests
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Audit Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    [data-testid="stSidebar"] * { color: #e8eaf0 !important; }
    [data-testid="stMetric"] {
        background-color: #22263a;
        border: 1px solid #2e3250;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #7a7f9a !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #e8eaf0 !important; }
    h1, h2, h3 { color: #e8eaf0 !important; }
    p, li { color: #c0c4d8; }
    hr { border-color: #2e3250; }
    [data-testid="stFileUploader"] {
        background-color: #1a1d27;
        border: 1px dashed #2e3250;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d27; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #7a7f9a !important; }
    .stTabs [aria-selected="true"]    { color: #4f8ef7 !important; border-bottom: 2px solid #4f8ef7; }
    .stRadio label { color: #e8eaf0 !important; }
    .stButton > button {
        background-color: #4f8ef7; color: white;
        border: none; border-radius: 6px; font-weight: 600;
    }
    .stButton > button:hover { background-color: #3a7de0; }
    .info-card {
        background-color: #22263a;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-label { color: #7a7f9a; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; }
    .info-value { color: #e8eaf0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────
BG_DARK  = "#0f1117"
BG_CARD  = "#22263a"
BORDER   = "#2e3250"
TXT_MUTE = "#7a7f9a"
TXT_MAIN = "#e8eaf0"

def dark_fig(w=11, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TXT_MUTE, labelsize=8)
    ax.xaxis.label.set_color(TXT_MUTE)
    ax.yaxis.label.set_color(TXT_MUTE)
    ax.title.set_color(TXT_MAIN)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    fig.tight_layout(pad=2)
    return fig, ax

def style_ax(ax):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TXT_MUTE, labelsize=8)
    ax.xaxis.label.set_color(TXT_MUTE)
    ax.yaxis.label.set_color(TXT_MUTE)
    ax.title.set_color(TXT_MAIN)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# MASTER SHEET SCRAPER
# ─────────────────────────────────────────────────────────────────
def get_master_sheet_info(fileobj):
    try:
        ms = pd.read_excel(fileobj, sheet_name="Master Sheet", header=None)

        def safe_get(row, col):
            try:
                val = ms.iloc[row, col]
                return str(val).strip() if pd.notna(val) else None
            except Exception:
                return None

        info = {
            "account"        : safe_get(0, 6),
            "customer_name"  : safe_get(1, 6),
            "own_rent"       : safe_get(2, 6),
            "community"      : safe_get(3, 6),
            "address"        : safe_get(4, 6),
            "city_town"      : safe_get(5, 6),
            "gru_rep"        : safe_get(6, 2),
            "survey_date"    : safe_get(7, 2),
            "survey_time"    : safe_get(8, 2),
            "results_sent_to": safe_get(9, 2),
        }

        if info["survey_date"] and "00:00:00" in info["survey_date"]:
            try:
                info["survey_date"] = pd.to_datetime(info["survey_date"]).strftime("%m/%d/%Y")
            except Exception:
                pass

        return info
    except Exception:
        return {}


def render_customer_card(info):
    if not info:
        return
    name      = info.get("customer_name",  "Unknown")
    account   = info.get("account",        "—")
    address   = info.get("address",        "—")
    city      = info.get("city_town",      "—")
    own_rent  = info.get("own_rent",       "—")
    community = info.get("community",      "—")
    rep       = info.get("gru_rep",        "—")
    date      = info.get("survey_date",    "—")
    time_     = info.get("survey_time",    "—")
    email     = info.get("results_sent_to","—")

    st.markdown(f"""
    <div class="info-card">
        <div style="font-size:1.2rem; font-weight:700; color:#e8eaf0; margin-bottom:10px;">
            👤 {name} &nbsp;&nbsp;
            <span style="color:#4f8ef7; font-size:0.95rem;">Account # {account}</span>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px;">
            <div><div class="info-label">Address</div><div class="info-value">{address}</div></div>
            <div><div class="info-label">City</div><div class="info-value">{city}</div></div>
            <div><div class="info-label">Own / Rent</div><div class="info-value">{own_rent}</div></div>
            <div><div class="info-label">Community</div><div class="info-value">{community}</div></div>
            <div><div class="info-label"> Rep</div><div class="info-value">{rep}</div></div>
            <div><div class="info-label">Survey Date</div><div class="info-value">{date} {time_}</div></div>
            <div><div class="info-label">Results Sent To</div><div class="info-value">{email}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TEMPERATURE
# ─────────────────────────────────────────────────────────────────
GAINESVILLE_LAT = 29.6516
GAINESVILLE_LON = -82.3248

@st.cache_data(ttl=3600)
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
    except Exception:
        return None


def merge_consumption_temp(df_div, df_temp):
    """Merge meter reading data with temperature — averages temp across each billing period."""
    df = df_div[df_div["consumption"] > 0].copy().sort_values("mr_date").reset_index(drop=True)
    temp_avgs, temp_maxs, temp_mins = [], [], []
    for _, row in df.iterrows():
        end_date   = row["mr_date"]
        start_date = end_date - pd.Timedelta(days=int(row["days"]))
        mask       = (df_temp.index >= start_date) & (df_temp.index <= end_date)
        period     = df_temp[mask]
        if not period.empty:
            temp_avgs.append(period["temp_avg"].mean())
            temp_maxs.append(period["temp_max"].mean())
            temp_mins.append(period["temp_min"].mean())
        else:
            temp_avgs.append(None)
            temp_maxs.append(None)
            temp_mins.append(None)
    df["temp_avg"] = temp_avgs
    df["temp_max"] = temp_maxs
    df["temp_min"] = temp_mins
    return df.dropna(subset=["temp_avg"])


def merge_ami_temp(daily_series, df_temp):
    """Merge AMI daily totals with daily temperature."""
    df = daily_series.reset_index()
    df.columns = ["date", "kwh"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(
        df_temp.reset_index().rename(columns={"date": "date"}),
        on="date", how="inner"
    )
    return df.dropna(subset=["temp_avg"])


def render_temp_charts(df_merged, x_col, y_col, unit, title_prefix, tab5, tab6, tab7):
    """Shared function to render all 3 temperature charts for both meter and AMI pages."""

    with tab5:
        fig, ax1 = plt.subplots(figsize=(11, 3.8))
        fig.patch.set_facecolor(BG_DARK)
        style_ax(ax1)
        ax1.bar(df_merged[x_col], df_merged[y_col],
                width=20 if x_col == "mr_date" else 0.8,
                color="#4f8ef7", alpha=0.6, label=unit)
        ax1.set_ylabel(unit, color="#4f8ef7")
        ax1.tick_params(axis="y", labelcolor="#4f8ef7")
        ax1.tick_params(axis="x", colors=TXT_MUTE)
        ax2 = ax1.twinx()
        ax2.plot(df_merged[x_col], df_merged["temp_avg"],
                 color="#f76f6f", linewidth=2.2, marker="o", markersize=4, label="Avg Temp °F")
        ax2.fill_between(df_merged[x_col],
                         df_merged["temp_min"], df_merged["temp_max"],
                         color="#f76f6f", alpha=0.08)
        ax2.set_ylabel("Temperature (°F)", color="#f76f6f")
        ax2.tick_params(axis="y", labelcolor="#f76f6f")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2,
                   facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
        ax1.set_title(f"{title_prefix} — Usage vs Temperature (Overlay)", color=TXT_MAIN)
        fig.autofmt_xdate()
        fig.tight_layout(pad=2)
        show_fig(fig)

    with tab6:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        fig.patch.set_facecolor(BG_DARK)
        style_ax(ax1)
        style_ax(ax2)
        ax1.bar(df_merged[x_col], df_merged[y_col],
                width=20 if x_col == "mr_date" else 0.8,
                color="#4f8ef7", alpha=0.85)
        ax1.set_ylabel(unit, color=TXT_MUTE)
        ax1.set_title(f"{title_prefix} — Usage", color=TXT_MAIN)
        ax2.plot(df_merged[x_col], df_merged["temp_avg"],
                 color="#f76f6f", linewidth=2, marker="o", markersize=4, label="Avg Temp")
        ax2.fill_between(df_merged[x_col],
                         df_merged["temp_min"], df_merged["temp_max"],
                         color="#f76f6f", alpha=0.1, label="Min–Max Range")
        ax2.axhline(65, color=TXT_MUTE, linewidth=1, linestyle="--", label="65°F baseline")
        ax2.set_ylabel("Temperature (°F)", color=TXT_MUTE)
        ax2.set_title("Gainesville FL — Avg Temperature", color=TXT_MAIN)
        ax2.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout(pad=2)
        show_fig(fig)

    with tab7:
        corr = df_merged[y_col].corr(df_merged["temp_avg"])
        fig, ax = dark_fig(7, 5)
        ax.scatter(df_merged["temp_avg"], df_merged[y_col],
                   color="#b06ef7", alpha=0.75, edgecolors="white", s=60)
        z      = np.polyfit(df_merged["temp_avg"], df_merged[y_col], 1)
        p      = np.poly1d(z)
        x_line = np.linspace(df_merged["temp_avg"].min(), df_merged["temp_avg"].max(), 100)
        ax.plot(x_line, p(x_line), color="darkorange", linewidth=2,
                linestyle="--", label="Trend")
        ax.set_title(f"{title_prefix} — Usage vs Temp  (r = {corr:.2f})", color=TXT_MAIN)
        ax.set_xlabel("Avg Temperature (°F)")
        ax.set_ylabel(unit)
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
        show_fig(fig)

        if corr > 0.6:
            st.info(f"r = {corr:.2f} — Strong positive correlation. Usage rises significantly with temperature.")
        elif corr > 0.3:
            st.info(f"r = {corr:.2f} — Moderate correlation. Temperature has some influence on usage.")
        elif corr < -0.3:
            st.info(f"r = {corr:.2f} — Negative correlation. Usage rises when temperature drops (heating load).")
        else:
            st.info(f"r = {corr:.2f} — Weak correlation. Usage appears mostly independent of temperature.")


# ─────────────────────────────────────────────────────────────────
# METER LOADER
# ─────────────────────────────────────────────────────────────────
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
        self.fileobj = fileobj
        self.df      = None

    def _find_sheet(self, xl):
        for name in xl.sheet_names:
            if "consumption" in name.lower():
                return name
        raise ValueError(f"No consumption sheet found. Sheets: {xl.sheet_names}")

    def load_and_clean(self):
        xl    = pd.ExcelFile(self.fileobj)
        sheet = self._find_sheet(xl)
        df    = pd.read_excel(xl, sheet_name=sheet, header=0)
        df.columns = df.columns.str.strip()
        df = df.rename(columns=self.COLUMN_MAP)

        df["mr_date"] = pd.to_datetime(df["mr_date"], errors="coerce")

        if df["consumption"].dtype == object:
            df["consumption"] = df["consumption"].astype(str).str.replace(",", "", regex=False)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

        for col in ["mr_result", "days", "avg_daily"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["mr_date"])
        df = df[~df["mr_reason"].isin(self.NON_READ_REASONS)]
        df = df[df["days"] > 0]
        df = df[(df["consumption"] > 0) | (df["mr_reason"].isin(self.VLINE_REASONS))]
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

    def get_customer_info(self):
        try:
            ms      = pd.read_excel(self.fileobj, sheet_name="Master Sheet", header=None)
            account = str(ms.iloc[0, 6]).strip()
            name    = str(ms.iloc[1, 6]).strip()
            return name, account
        except Exception:
            return "Unknown", "Unknown"


# ─────────────────────────────────────────────────────────────────
# METER FEATURES
# ─────────────────────────────────────────────────────────────────
class MeterFeatures:
    def __init__(self, df):
        self.df = df.copy().sort_values("mr_date").reset_index(drop=True)

    def compute_features(self):
        df     = self.df
        df_act = df[df["consumption"] > 0]

        avg_read_interval = df_act["days"].mean()
        total_consumption = df_act["consumption"].sum()
        total_days        = df_act["days"].sum()
        overall_daily_avg = total_consumption / total_days if total_days > 0 else None
        peak_consumption  = df_act["consumption"].max()
        base_consumption  = df_act["consumption"].quantile(0.05)
        period_series     = df_act.set_index("mr_date")["consumption"]
        rolling_avg       = period_series.rolling(window=3).mean()

        daily_avg_series = None
        if "avg_daily" in df_act.columns:
            s = df_act.set_index("mr_date")["avg_daily"].replace(0, np.nan).dropna()
            if not s.empty:
                daily_avg_series = s

        iso_cols = [c for c in ["consumption", "days", "avg_daily"] if c in df_act.columns]
        iso_data = df_act[iso_cols].dropna()
        df["anomaly"] = False
        if len(iso_data) >= 5:
            preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(iso_data)
            df.loc[iso_data.index, "anomaly"] = (preds == -1)

        unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""

        return {
            "avg_read_interval": avg_read_interval,
            "total_consumption": total_consumption,
            "overall_daily_avg": overall_daily_avg,
            "peak_consumption" : peak_consumption,
            "base_consumption" : base_consumption,
            "period_series"    : period_series,
            "rolling_avg"      : rolling_avg,
            "daily_avg_series" : daily_avg_series,
            "df_with_anomalies": df,
            "n_anomalies"      : int(df["anomaly"].sum()),
            "unit"             : unit,
        }


# ─────────────────────────────────────────────────────────────────
# MR REASON MARKERS
# ─────────────────────────────────────────────────────────────────
def get_meter_changes(df):
    df      = df.sort_values("mr_date")
    reasons = df["mr_reason"].tolist()
    dates   = df["mr_date"].tolist()
    changes = []
    i = 0
    while i < len(reasons):
        if reasons[i] == 22:
            for j in range(i+1, min(i+4, len(reasons))):
                if reasons[j] == 21:
                    changes.append((dates[i], dates[j]))
                    i = j + 1
                    break
            else:
                changes.append((dates[i], dates[i]))
                i += 1
        elif reasons[i] == 21:
            found = any(reasons[k] == 22 for k in range(max(0, i-4), i))
            if not found:
                changes.append((dates[i], dates[i]))
            i += 1
        else:
            i += 1
    return changes


def add_markers(ax, df):
    move_ins    = df[df["mr_reason"] == 6]
    first_label = True
    for _, row in move_ins.iterrows():
        lbl = "Move-In" if first_label else "_nolegend_"
        ax.axvline(x=row["mr_date"], color="dodgerblue",
                   linewidth=1.8, linestyle="--", alpha=0.9, label=lbl)
        first_label = False

    changes     = get_meter_changes(df)
    first_label = True
    for date_start, date_end in changes:
        lbl = "Meter Change" if first_label else "_nolegend_"
        if date_start == date_end:
            ax.axvline(x=date_start, color="darkorange",
                       linewidth=1.8, linestyle="--", alpha=0.9, label=lbl)
        else:
            ax.axvspan(date_start, date_end, color="darkorange", alpha=0.18, label=lbl)
            ax.axvline(x=date_start, color="darkorange", linewidth=1.2, linestyle="--", alpha=0.6)
            ax.axvline(x=date_end,   color="darkorange", linewidth=1.2, linestyle="--", alpha=0.6)
        first_label = False


# ─────────────────────────────────────────────────────────────────
# AMI LOADER
# ─────────────────────────────────────────────────────────────────
class AMILoader:
    ELECTRIC_SHEET_NAMES = ["ELECTRIC", "Electric", "Sheet1", "SHEET1"]

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.df      = None

    def _find_sheet(self, xl):
        for name in xl.sheet_names:
            if name in self.ELECTRIC_SHEET_NAMES:
                return name
        return xl.sheet_names[0]

    def load_and_clean(self):
        xl    = pd.ExcelFile(self.fileobj)
        sheet = self._find_sheet(xl)
        df    = pd.read_excel(xl, sheet_name=sheet, header=None, skiprows=4)
        df    = df[[0, 1]].copy()
        df.columns = ["timestamp", "wh_raw"]

        df["timestamp"] = (df["timestamp"].astype(str)
                           .str.replace(r"\s+EST.*$", "", regex=True)
                           .str.replace(r"\s+EDT.*$", "", regex=True)
                           .str.strip())
        df["timestamp"] = pd.to_datetime(df["timestamp"],
                                         format="%b %d, %Y - %I:%M %p",
                                         errors="coerce")
        df["wh"]  = (df["wh_raw"].astype(str)
                     .str.replace(",", "", regex=False)
                     .str.extract(r"([\d.]+)")[0])
        df["wh"]  = pd.to_numeric(df["wh"],  errors="coerce")
        df["kwh"] = df["wh"] / 1000

        df = df.dropna(subset=["timestamp", "kwh"])
        df = df[df["kwh"] > 0]
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        return df


# ─────────────────────────────────────────────────────────────────
# AMI FEATURES
# ─────────────────────────────────────────────────────────────────
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
        peak_day         = daily_series.idxmax()
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


# ─────────────────────────────────────────────────────────────────
# YEAR-OVER-YEAR
# ─────────────────────────────────────────────────────────────────
def compute_yoy(df_div):
    df_div  = df_div[df_div["consumption"] > 0].sort_values("mr_date")
    latest  = df_div["mr_date"].max()
    cut_1yr = latest - pd.DateOffset(years=1)
    cut_2yr = latest - pd.DateOffset(years=2)
    recent  = df_div[df_div["mr_date"] >  cut_1yr]
    prior   = df_div[(df_div["mr_date"] > cut_2yr) & (df_div["mr_date"] <= cut_1yr)]
    if recent.empty or prior.empty:
        return None
    recent_avg = recent["consumption"].sum() / recent["days"].sum()
    prior_avg  = prior["consumption"].sum()  / prior["days"].sum()
    pct        = ((recent_avg - prior_avg) / prior_avg) * 100 if prior_avg else None
    return {
        "recent_avg": round(recent_avg, 2),
        "prior_avg" : round(prior_avg, 2),
        "pct_change": round(pct, 1) if pct else None,
        "trend"     : "▲ UP" if pct and pct > 10 else ("▼ DOWN" if pct and pct < -10 else "● STABLE"),
        "recent_df" : recent,
        "prior_df"  : prior,
    }


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Energy Audit")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊  Single File Analysis",
        "📈  Year-over-Year",
        "⚡  AMI Analysis",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Daniel Quiroz")
    st.caption("Built for the Energy Audit Team")


# ─────────────────────────────────────────────────────────────────
# PAGE: SINGLE FILE ANALYSIS
# ─────────────────────────────────────────────────────────────────
if page == "📊  Single File Analysis":
    st.title("📊 Single File Analysis")

    uploaded = st.file_uploader("Upload Customer Meter File (.xlsx)", type=["xlsx"], key="meter")

    if uploaded:
        file_bytes = uploaded.read()
        try:
            info = get_master_sheet_info(io.BytesIO(file_bytes))
            render_customer_card(info)
            st.markdown("---")

            loader = MeterLoader(io.BytesIO(file_bytes))
            loader.load_and_clean()

            division = st.radio("Division", ["Electricity", "Water"], horizontal=True)
            df_div   = loader.get_division(division)

            if df_div.empty:
                st.warning(f"No {division} data found in this file.")
            else:
                feats   = MeterFeatures(df_div).compute_features()
                unit    = feats["unit"]
                df_full = feats["df_with_anomalies"]

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Consumption", f"{feats['total_consumption']:,.1f} {unit}")
                c2.metric("Daily Average",
                          f"{feats['overall_daily_avg']:.2f} {unit}/day"
                          if feats["overall_daily_avg"] else "N/A")
                c3.metric("Peak Period",    f"{feats['peak_consumption']:,.1f} {unit}")
                c4.metric("Base Load (p5)", f"{feats['base_consumption']:,.1f} {unit}")
                c5.metric("Anomalies",      str(feats["n_anomalies"]),
                          delta="flagged" if feats["n_anomalies"] > 0 else None,
                          delta_color="inverse")
                st.markdown("---")

                t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                    "Consumption", "Daily Avg", "Rolling Avg", "Anomalies",
                    "🌡 Temp Overlay", "🌡 Side-by-Side", "🌡 Scatter"
                ])

                with t1:
                    fig, ax = dark_fig()
                    s = feats["period_series"]
                    ax.bar(s.index, s.values, width=20, color="#4f8ef7", alpha=0.85, label="Consumption")
                    add_markers(ax, df_full)
                    ax.set_title(f"{division} — Consumption per Read Period")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                    fig.autofmt_xdate()
                    show_fig(fig)

                with t2:
                    fig, ax = dark_fig()
                    s = feats["daily_avg_series"]
                    if s is not None and not s.empty:
                        ax.plot(s.index, s.values, color="#f7c94f",
                                linewidth=2, marker="o", markersize=4, label="Daily Avg")
                        add_markers(ax, df_full)
                        ax.set_title(f"{division} — Average Daily Usage per Period")
                        ax.set_ylabel(f"{unit}/day")
                        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                        fig.autofmt_xdate()
                    else:
                        ax.text(0.5, 0.5, "No daily avg data available",
                                ha="center", va="center", color=TXT_MUTE, transform=ax.transAxes)
                    show_fig(fig)

                with t3:
                    fig, ax = dark_fig()
                    s = feats["period_series"]
                    r = feats["rolling_avg"]
                    ax.plot(s.index, s.values, color="#4f8ef7", alpha=0.4,
                            linewidth=1.5, label="Consumption")
                    ax.plot(r.index, r.values, color="#f76f6f",
                            linewidth=2.5, label="3-Read Rolling Avg")
                    add_markers(ax, df_full)
                    ax.set_title(f"{division} — Consumption Trend")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                    fig.autofmt_xdate()
                    show_fig(fig)

                with t4:
                    fig, ax = dark_fig()
                    df_a    = feats["df_with_anomalies"]
                    normal  = df_a[(~df_a["anomaly"]) & (df_a["consumption"] > 0)]
                    anomaly = df_a[ df_a["anomaly"]   & (df_a["consumption"] > 0)]
                    ax.bar(normal["mr_date"],  normal["consumption"],
                           width=20, color="#4f8ef7", alpha=0.85, label="Normal")
                    ax.bar(anomaly["mr_date"], anomaly["consumption"],
                           width=20, color="#f76f6f", alpha=0.9,  label="Anomaly")
                    add_markers(ax, df_full)
                    ax.set_title(f"{division} — Anomaly Detection")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                    fig.autofmt_xdate()
                    show_fig(fig)
                    if not anomaly.empty:
                        st.warning(f"⚠ {len(anomaly)} anomalous read period(s) detected:")
                        cols = [c for c in ["mr_date","mr_reason","days","consumption","avg_daily"]
                                if c in anomaly.columns]
                        st.dataframe(anomaly[cols], use_container_width=True)

                # Temperature charts
                with st.spinner("Fetching Gainesville temperature data..."):
                    df_temp = get_gainesville_temps(
                        str(df_div["mr_date"].min().date()),
                        str(df_div["mr_date"].max().date())
                    )

                if df_temp is not None:
                    df_merged = merge_consumption_temp(df_div, df_temp)
                    name      = info.get("customer_name", division)
                    render_temp_charts(df_merged, "mr_date", "consumption",
                                       unit, f"{name} — {division}", t5, t6, t7)
                else:
                    for tab in [t5, t6, t7]:
                        with tab:
                            st.warning("Could not fetch temperature data. Check internet connection.")

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# PAGE: YEAR-OVER-YEAR
# ─────────────────────────────────────────────────────────────────
elif page == "📈  Year-over-Year":
    st.title("📈 Year-over-Year Comparison")

    uploaded = st.file_uploader("Upload Customer Meter File (.xlsx)", type=["xlsx"], key="yoy")

    if uploaded:
        file_bytes = uploaded.read()
        try:
            info = get_master_sheet_info(io.BytesIO(file_bytes))
            render_customer_card(info)
            st.markdown("---")

            loader = MeterLoader(io.BytesIO(file_bytes))
            loader.load_and_clean()

            division = st.radio("Division", ["Electricity", "Water"], horizontal=True)
            df_div   = loader.get_division(division)

            if df_div.empty:
                st.warning(f"No {division} data found.")
            else:
                yoy  = compute_yoy(df_div)
                unit = df_div["mr_unit"].iloc[0] if "mr_unit" in df_div.columns else ""

                if yoy is None:
                    st.warning("Need at least 2 years of data for year-over-year comparison.")
                else:
                    pct = yoy["pct_change"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prior 12mo Daily Avg",  f"{yoy['prior_avg']:.2f} {unit}/day")
                    c2.metric("Recent 12mo Daily Avg", f"{yoy['recent_avg']:.2f} {unit}/day")
                    c3.metric("% Change",
                              f"{pct:+.1f}%" if pct else "N/A",
                              delta=f"{pct:+.1f}%" if pct else None,
                              delta_color="inverse")
                    c4.metric("Trend", yoy["trend"])
                    st.markdown("---")

                    fig, ax = dark_fig(11, 4.2)
                    prior  = yoy["prior_df"].set_index("mr_date")["consumption"]
                    recent = yoy["recent_df"].set_index("mr_date")["consumption"]
                    ax.plot(prior.index,  prior.values,  color=TXT_MUTE, linewidth=2,
                            marker="o", markersize=4, linestyle="--", label="Prior 12 months")
                    ax.plot(recent.index, recent.values, color="#4f8ef7", linewidth=2.5,
                            marker="o", markersize=5, label="Recent 12 months")
                    ax.set_title(f"{division} — Year-over-Year Comparison")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=9)
                    fig.autofmt_xdate()
                    show_fig(fig)

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# PAGE: AMI ANALYSIS
# ─────────────────────────────────────────────────────────────────
elif page == "⚡  AMI Analysis":
    st.title("⚡ AMI Analysis")
    st.markdown("Upload a 15-minute interval AMI file.")

    uploaded = st.file_uploader("Upload AMI Excel File (.xlsx)", type=["xlsx"], key="ami")

    if uploaded:
        try:
            loader    = AMILoader(uploaded)
            df_ami    = loader.load_and_clean()
            ami_feats = AMIFeatures(df_ami).compute()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Interval",       f"{ami_feats['interval_minutes']} min")
            c2.metric("Base Load",      f"{ami_feats['base_load']:.3f} kWh",
                      help="5th percentile — steady background draw")
            c3.metric("Base Load (kW)", f"{ami_feats['base_load_kw']:.3f} kW")
            c4.metric("Peak Demand",    f"{ami_feats['peak_kw']:.3f} kW")
            c5.metric("Daily Avg",      f"{ami_feats['daily_avg_kwh']:.2f} kWh/day")
            st.markdown("---")

            t1, t2, t3, t4, t5, t6 = st.tabs([
                "Load Shape", "Daily Totals", "Hourly Profile",
                "🌡 Temp Overlay", "🌡 Side-by-Side", "🌡 Scatter"
            ])

            with t1:
                fig, ax = dark_fig(12, 4)
                df_p    = ami_feats["df"]
                ax.plot(df_p["timestamp"], df_p["kwh"],
                        color="#4f8ef7", linewidth=0.6, alpha=0.85)
                ax.axhline(ami_feats["base_load"], color="#3ecf8e", linewidth=1.5,
                           linestyle="--", label=f"Base Load ({ami_feats['base_load']:.3f} kWh)")
                ax.axhline(ami_feats["peak_kwh"],  color="#f76f6f", linewidth=1.5,
                           linestyle="--", label=f"Peak ({ami_feats['peak_kwh']:.3f} kWh)")
                ax.set_title("AMI Load Shape — 15-Minute Intervals")
                ax.set_ylabel("kWh")
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                fig.autofmt_xdate()
                show_fig(fig)

            with t2:
                fig, ax  = dark_fig(12, 4)
                ds       = ami_feats["daily_series"]
                peak_day = ami_feats["peak_day"]
                colors   = ["#f76f6f" if d == peak_day else "#4f8ef7" for d in ds.index]
                ax.bar(ds.index, ds.values, color=colors, alpha=0.85, width=0.8)
                ax.axhline(ami_feats["daily_avg_kwh"], color="#f7c94f", linewidth=1.8,
                           linestyle="--",
                           label=f"Daily Avg ({ami_feats['daily_avg_kwh']:.2f} kWh)")
                ax.set_title(f"Daily Total Usage — red = peak day ({peak_day})")
                ax.set_ylabel("kWh / day")
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                fig.autofmt_xdate()
                show_fig(fig)

            with t3:
                fig, ax = dark_fig(10, 4)
                ah      = ami_feats["avg_by_hour"]
                ax.bar(ah.index, ah.values, color="#b06ef7", alpha=0.85, width=0.7)
                ax.axhline(ami_feats["base_load"], color="#3ecf8e", linewidth=1.5,
                           linestyle="--",
                           label=f"Base Load ({ami_feats['base_load']:.3f} kWh)")
                ax.set_title("Average Load Shape by Hour of Day")
                ax.set_xlabel("Hour (0 = midnight)")
                ax.set_ylabel("Avg kWh per interval")
                ax.set_xticks(range(0, 24))
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TXT_MAIN, fontsize=8)
                show_fig(fig)

            # Temperature charts for AMI
            with st.spinner("Fetching Gainesville temperature data..."):
                start   = pd.Timestamp(ami_feats["df"]["date"].min())
                end     = pd.Timestamp(ami_feats["df"]["date"].max())
                df_temp = get_gainesville_temps(
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d")
                )

            if df_temp is not None:
                # Merge AMI daily totals with temperature
                df_ami_temp = merge_ami_temp(ami_feats["daily_series"], df_temp)
                render_temp_charts(df_ami_temp, "date", "kwh",
                                   "kWh/day", "AMI", t4, t5, t6)
            else:
                for tab in [t4, t5, t6]:
                    with tab:
                        st.warning("Could not fetch temperature data. Check internet connection.")

        except Exception as e:
            st.error(f"Error loading AMI file: {e}")
