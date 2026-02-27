# GRU Energy Audit Analyzer

A Streamlit web app for analyzing GRU customer meter reading and AMI data.

## Features
- **Single File Analysis** — Load a customer Excel file, view consumption, daily averages, rolling trends, and anomaly detection for Electricity or Water
- **Year-over-Year** — Compare last 12 months vs prior 12 months with % change
- **AMI Analysis** — Load a 15-minute interval AMI file, view load shape, daily totals, peak demand, base load, and hourly profile

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy**
5. Share the URL with your team

## File Formats Supported

**Meter Files:** GRU customer Excel files with a `Consumption` or `Consumption History` sheet  
**AMI Files:** GRU AMI Excel files with 15-minute interval data (Wh format)
