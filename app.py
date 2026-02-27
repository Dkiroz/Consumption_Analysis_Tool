"""
GRU Energy Audit Analyzer — Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.ensemble import IsolationForest
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRU Energy Audit Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    [data-testid="stSidebar"] * { color: #e8eaf0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #22263a;
        border: 1px solid #2e3250;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { color: #7a7f9a !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"]  { color: #e8eaf0 !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

    /* Headers */
    h1, h2, h3 { color: #e8eaf0 !important; }
    p, li       { color: #c0c4d8; }

    /* Divider */
    hr { border-color: #2e3250; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1d27;
        border: 1px dashed #2e3250;
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d27; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #7a7f9a !important; }
    .stTabs [aria-selected="true"]    { color: #4f8ef7 !important; border-bottom: 2px solid #4f8ef7; }

    /* Radio buttons */
    .stRadio label { color: #e8eaf0 !important; }

    /* Buttons */
    .stButton > button {
        background-color: #4f8ef7;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #3a7de0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK STYLE HELPER
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

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# METER LOADER
# ─────────────────────────────────────────────────────────────────
class MeterLoader:
    COLUMN_MAP = {
        "Division": "division", "Device": "device",
        "MR Reason": "mr_reason", "MR Type": "mr_type",
        "MR Date": "mr_date", "Days": "days",
        "MR Result": "mr_result", "MR Unit": "mr_unit",
        "Consumption": "consumption", "Avg.": "avg_daily", "Avg": "avg_daily",
    }
    NON_READ_REASONS = {3, 6, 13, 18, 21}

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.df = None

    def _find_sheet(self, xl):
        for name in xl.sheet_names:
            if "consumption" in name.lower():
                return name
        raise ValueError(f"No consumption sheet found. Sheets: {xl.sheet_names}")

    def load_and_clean(self):
        xl    = pd.ExcelFile(self.fileobj)
        sheet = self._find_sheet(xl)
        df    = pd.read_excel(xl, sheet_name=sheet, header=0)
        df    = df.rename(columns=self.COLUMN_MAP)

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
        df = df[df["consumption"] > 0]
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

        unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""
        return {
            "avg_read_interval": avg_read_interval,
            "total_consumption": total_consumption,
            "overall_daily_avg": overall_daily_avg,
            "peak_consumption":  peak_consumption,
            "base_consumption":  base_consumption,
            "period_series":     period_series,
            "rolling_avg":       rolling_avg,
            "daily_avg_series":  daily_avg_series,
            "df_with_anomalies": df,
            "n_anomalies":       int(df["anomaly"].sum()),
            "unit":              unit,
        }


# ─────────────────────────────────────────────────────────────────
# AMI LOADER
# ─────────────────────────────────────────────────────────────────
class AMILoader:
    ELECTRIC_SHEET_NAMES = ["ELECTRIC", "Electric", "Sheet1", "SHEET1"]

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.df = None

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

        df["wh"] = (df["wh_raw"].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.extract(r"([\d.]+)")[0])
        df["wh"]  = pd.to_numeric(df["wh"], errors="coerce")
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

        df["date"]    = df["timestamp"].dt.date
        daily_series  = df.groupby("date")["kwh"].sum()
        daily_avg_kwh = daily_series.mean()
        peak_day      = daily_series.idxmax()

        df["hour"]    = df["timestamp"].dt.hour
        avg_by_hour   = df.groupby("hour")["kwh"].mean()

        return {
            "interval_minutes": interval_minutes,
            "base_load":        base_load,
            "base_load_kw":     base_load_kw,
            "peak_kwh":         peak_kwh,
            "peak_kw":          peak_kw,
            "daily_avg_kwh":    daily_avg_kwh,
            "daily_series":     daily_series,
            "peak_day":         peak_day,
            "avg_by_hour":      avg_by_hour,
            "df":               df,
        }


# ─────────────────────────────────────────────────────────────────
# YEAR-OVER-YEAR
# ─────────────────────────────────────────────────────────────────
def compute_yoy(df_div):
    df_div  = df_div.sort_values("mr_date")
    latest  = df_div["mr_date"].max()
    cut_1yr = latest - pd.DateOffset(years=1)
    cut_2yr = latest - pd.DateOffset(years=2)
    recent  = df_div[df_div["mr_date"] > cut_1yr]
    prior   = df_div[(df_div["mr_date"] > cut_2yr) & (df_div["mr_date"] <= cut_1yr)]

    if recent.empty or prior.empty:
        return None

    recent_avg = recent["consumption"].sum() / recent["days"].sum()
    prior_avg  = prior["consumption"].sum()  / prior["days"].sum()
    pct        = ((recent_avg - prior_avg) / prior_avg) * 100 if prior_avg else None

    return {
        "recent_avg": round(recent_avg, 2),
        "prior_avg":  round(prior_avg, 2),
        "pct_change": round(pct, 1) if pct else None,
        "trend":      "▲ UP" if pct and pct > 10 else ("▼ DOWN" if pct and pct < -10 else "● STABLE"),
        "recent_df":  recent,
        "prior_df":   prior,
    }


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ GRU Energy Audit")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊  Single File Analysis",
        "📈  Year-over-Year",
        "⚡  AMI Analysis",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("GRU Energy & Water Savings Plan")
    st.caption("Built for the Energy Audit Team")


# ─────────────────────────────────────────────────────────────────
# PAGE: SINGLE FILE ANALYSIS
# ─────────────────────────────────────────────────────────────────
if page == "📊  Single File Analysis":
    st.title("📊 Single File Analysis")
    st.markdown("Upload a customer meter Excel file to analyze electricity and water usage.")

    uploaded = st.file_uploader("Upload Customer Meter File (.xlsx)",
                                type=["xlsx"], key="meter_upload")

    if uploaded:
        try:
            loader = MeterLoader(uploaded)
            loader.load_and_clean()
            name, account = loader.get_customer_info()

            # Customer info banner
            st.markdown(f"### 👤 {name}  —  Account `{account}`")
            st.markdown("---")

            division = st.radio("Division", ["Electricity", "Water"], horizontal=True)
            df_div   = loader.get_division(division)

            if df_div.empty:
                st.warning(f"No {division} data found in this file.")
            else:
                feats = MeterFeatures(df_div).compute_features()
                unit  = feats["unit"]

                # Metric cards
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Consumption",  f"{feats['total_consumption']:,.1f} {unit}")
                c2.metric("Daily Average",       f"{feats['overall_daily_avg']:.2f} {unit}/day")
                c3.metric("Peak Period",          f"{feats['peak_consumption']:,.1f} {unit}")
                c4.metric("Base Load (p5)",       f"{feats['base_consumption']:,.1f} {unit}")
                c5.metric("Anomalies",            str(feats["n_anomalies"]),
                          delta="flagged" if feats["n_anomalies"] > 0 else None,
                          delta_color="inverse")

                st.markdown("---")

                # Charts in tabs
                t1, t2, t3, t4 = st.tabs(["Consumption", "Daily Average", "Rolling Avg", "Anomalies"])

                with t1:
                    fig, ax = dark_fig()
                    s = feats["period_series"]
                    ax.bar(s.index, s.values, width=20, color="#4f8ef7", alpha=0.85)
                    ax.set_title(f"{division} — Consumption per Read Period")
                    ax.set_ylabel(unit)
                    fig.autofmt_xdate()
                    show_fig(fig)

                with t2:
                    fig, ax = dark_fig()
                    if feats["daily_avg_series"] is not None:
                        s = feats["daily_avg_series"]
                        ax.plot(s.index, s.values, color="#f7c94f",
                                linewidth=2, marker="o", markersize=4)
                        ax.set_title(f"{division} — Average Daily Usage per Period")
                        ax.set_ylabel(f"{unit}/day")
                        fig.autofmt_xdate()
                    else:
                        ax.text(0.5, 0.5, "No daily avg data", ha="center",
                                va="center", color=TXT_MUTE, transform=ax.transAxes)
                    show_fig(fig)

                with t3:
                    fig, ax = dark_fig()
                    s = feats["period_series"]
                    r = feats["rolling_avg"]
                    ax.plot(s.index, s.values, color="#4f8ef7", alpha=0.4,
                            linewidth=1.5, label="Consumption")
                    ax.plot(r.index, r.values, color="#f76f6f",
                            linewidth=2.5, label="3-Read Rolling Avg")
                    ax.set_title(f"{division} — Consumption Trend")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                              labelcolor=TXT_MAIN, fontsize=8)
                    fig.autofmt_xdate()
                    show_fig(fig)

                with t4:
                    fig, ax = dark_fig()
                    df_a    = feats["df_with_anomalies"]
                    normal  = df_a[~df_a["anomaly"]]
                    anomaly = df_a[df_a["anomaly"]]
                    ax.bar(normal["mr_date"],  normal["consumption"],
                           width=20, color="#4f8ef7", alpha=0.85, label="Normal")
                    ax.bar(anomaly["mr_date"], anomaly["consumption"],
                           width=20, color="#f76f6f", alpha=0.9,  label="Anomaly")
                    ax.set_title(f"{division} — Anomaly Detection")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                              labelcolor=TXT_MAIN, fontsize=8)
                    fig.autofmt_xdate()
                    show_fig(fig)

                    if not anomaly.empty:
                        st.warning(f"⚠ {len(anomaly)} anomalous read period(s) detected:")
                        cols = [c for c in ["mr_date","days","consumption","avg_daily","mr_type"]
                                if c in anomaly.columns]
                        st.dataframe(anomaly[cols], use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")


# ─────────────────────────────────────────────────────────────────
# PAGE: YEAR-OVER-YEAR
# ─────────────────────────────────────────────────────────────────
elif page == "📈  Year-over-Year":
    st.title("📈 Year-over-Year Comparison")
    st.markdown("Compare the last 12 months of usage against the prior 12 months.")

    uploaded = st.file_uploader("Upload Customer Meter File (.xlsx)",
                                type=["xlsx"], key="yoy_upload")

    if uploaded:
        try:
            loader = MeterLoader(uploaded)
            loader.load_and_clean()
            name, account = loader.get_customer_info()
            st.markdown(f"### 👤 {name}  —  Account `{account}`")
            st.markdown("---")

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
                    # Metric cards
                    pct   = yoy["pct_change"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prior 12mo Daily Avg",  f"{yoy['prior_avg']:.2f} {unit}/day")
                    c2.metric("Recent 12mo Daily Avg", f"{yoy['recent_avg']:.2f} {unit}/day")
                    c3.metric("% Change",
                              f"{pct:+.1f}%" if pct else "N/A",
                              delta=f"{pct:+.1f}%" if pct else None,
                              delta_color="inverse")
                    c4.metric("Trend", yoy["trend"])

                    st.markdown("---")

                    # Chart
                    fig, ax = dark_fig(11, 4.2)
                    prior  = yoy["prior_df"].set_index("mr_date")["consumption"]
                    recent = yoy["recent_df"].set_index("mr_date")["consumption"]
                    ax.plot(prior.index,  prior.values,  color=TXT_MUTE, linewidth=2,
                            marker="o", markersize=4, linestyle="--",
                            label="Prior 12 months")
                    ax.plot(recent.index, recent.values, color="#4f8ef7", linewidth=2.5,
                            marker="o", markersize=5, label="Recent 12 months")
                    ax.set_title(f"{division} — Year-over-Year Comparison")
                    ax.set_ylabel(unit)
                    ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                              labelcolor=TXT_MAIN, fontsize=9)
                    fig.autofmt_xdate()
                    show_fig(fig)

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# PAGE: AMI ANALYSIS
# ─────────────────────────────────────────────────────────────────
elif page == "⚡  AMI Analysis":
    st.title("⚡ AMI Analysis")
    st.markdown("Upload a 15-minute interval AMI file to analyze load shape, base load, and peak demand.")

    uploaded = st.file_uploader("Upload AMI Excel File (.xlsx)",
                                type=["xlsx"], key="ami_upload")

    if uploaded:
        try:
            loader    = AMILoader(uploaded)
            df_ami    = loader.load_and_clean()
            ami_feats = AMIFeatures(df_ami).compute()

            # Metric cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Interval",       f"{ami_feats['interval_minutes']} min")
            c2.metric("Base Load",      f"{ami_feats['base_load']:.3f} kWh",
                      help="5th percentile — steady background draw")
            c3.metric("Base Load (kW)", f"{ami_feats['base_load_kw']:.3f} kW")
            c4.metric("Peak Demand",    f"{ami_feats['peak_kw']:.3f} kW")
            c5.metric("Daily Avg",      f"{ami_feats['daily_avg_kwh']:.2f} kWh/day")

            st.markdown("---")

            t1, t2, t3 = st.tabs(["Load Shape", "Daily Totals", "Hourly Profile"])

            with t1:
                st.markdown("Full 15-minute interval timeline with base load and peak lines.")
                fig, ax = dark_fig(12, 4)
                df_p    = ami_feats["df"]
                ax.plot(df_p["timestamp"], df_p["kwh"],
                        color="#4f8ef7", linewidth=0.6, alpha=0.85)
                ax.axhline(ami_feats["base_load"], color="#3ecf8e", linewidth=1.5,
                           linestyle="--",
                           label=f"Base Load ({ami_feats['base_load']:.3f} kWh)")
                ax.axhline(ami_feats["peak_kwh"], color="#f76f6f", linewidth=1.5,
                           linestyle="--",
                           label=f"Peak ({ami_feats['peak_kwh']:.3f} kWh)")
                ax.set_title("AMI Load Shape — 15-Minute Intervals")
                ax.set_ylabel("kWh")
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                          labelcolor=TXT_MAIN, fontsize=8)
                fig.autofmt_xdate()
                show_fig(fig)

            with t2:
                st.markdown(f"Daily totals — red bar = peak day ({ami_feats['peak_day']})")
                fig, ax  = dark_fig(12, 4)
                ds       = ami_feats["daily_series"]
                peak_day = ami_feats["peak_day"]
                colors   = ["#f76f6f" if d == peak_day else "#4f8ef7" for d in ds.index]
                ax.bar(ds.index, ds.values, color=colors, alpha=0.85, width=0.8)
                ax.axhline(ami_feats["daily_avg_kwh"], color="#f7c94f", linewidth=1.8,
                           linestyle="--",
                           label=f"Daily Avg ({ami_feats['daily_avg_kwh']:.2f} kWh)")
                ax.set_title("Daily Total Usage (kWh)")
                ax.set_ylabel("kWh / day")
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                          labelcolor=TXT_MAIN, fontsize=8)
                fig.autofmt_xdate()
                show_fig(fig)

            with t3:
                st.markdown("Average kWh per interval by hour — shows when the customer uses the most energy.")
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
                ax.legend(facecolor=BG_CARD, edgecolor=BORDER,
                          labelcolor=TXT_MAIN, fontsize=8)
                show_fig(fig)

        except Exception as e:
            st.error(f"Error loading AMI file: {e}")
