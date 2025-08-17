import streamlit as st
import pandas as pd
from io import BytesIO
from prophet import Prophet
import plotly.express as px
from pathlib import Path as _Path2
import json as _json2

def main():

    st.set_page_config(page_title="KISS Demand Forecasting", layout="wide")
    st.title("KISS Demand Forecasting")
    st.markdown("**Goal:** Reveal each stage: Load → Clean → Split → Train → Forecast → Export")

    st.sidebar.subheader("Preset (optional)")
    _tpl_dir = _Path2(__file__).resolve().parent.parent / "templates"
    _presets = [p.name for p in _tpl_dir.glob("*.json")]
    _chosen = st.sidebar.selectbox("Load preset", ["(none)"] + _presets)
    if _chosen != "(none)":
        with open(_tpl_dir / _chosen, "r", encoding="utf-8") as _f:
            _preset = _json2.load(_f)
        _defs = _preset.get("defaults", {})
        st.session_state.setdefault("preset_defaults", _defs)
        if st.sidebar.button("Apply preset defaults"):
            st.success(f"Applied preset defaults from {_chosen}")

    st.header("Stage 1 — Load Data")
    uploaded = st.file_uploader("Upload CSV (or leave empty to use built-in sample)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        _data_dir = _Path2(__file__).resolve().parent.parent / "data"
        _file_path = _data_dir / "synthetic_sales_dataset.csv"
        if _file_path.exists():
            df = pd.read_csv(_file_path)
        else:
            st.error("No data file found. Please upload a CSV or ensure the sample data is available.")
            return
    st.success(f"Loaded {len(df)} rows: {list(df.columns)}")
    if st.button("Show Raw Sample"):
        st.dataframe(df.head(20), use_container_width=True)

    st.header("Stage 2 — Clean & Select")
    df.columns=[c.strip() for c in df.columns]
    df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values(["SKU","Date"])
    skus=sorted(df["SKU"].unique())
    sku=st.selectbox("Choose SKU", skus, index=0)
    sku_df=df[df["SKU"]==sku][["Date","Sales_Qty"]].rename(columns={"Date":"ds","Sales_Qty":"y"}).dropna().reset_index(drop=True)
    if st.button("Show Cleaned Sample"):
        st.dataframe(sku_df.head(20), use_container_width=True)


    st.header("Stage 3 — Train/Test Split & Parameters")
    horizon_days = st.slider("Forecast Horizon (days)", 7, 180, st.session_state.get('preset_defaults',{}).get('horizon_days',60))
    conf = st.slider("Confidence Interval (%)", 50, 99, st.session_state.get('preset_defaults',{}).get('conf_pct',90))
    test_tail = st.slider("Hold-out Test Tail (days)", 0, min(60, len(sku_df)//3), st.session_state.get('preset_defaults',{}).get('test_tail_days',14))
    growth = st.selectbox("Trend Mode", ["linear","flat"], index=0 if st.session_state.get('preset_defaults',{}).get('trend_mode','linear')=='linear' else 1)
    if test_tail>0 and test_tail<len(sku_df):
        train=sku_df.iloc[:-test_tail].copy()
        test=sku_df.iloc[-test_tail:].copy()
    else:
        train=sku_df.copy(); test=pd.DataFrame(columns=sku_df.columns)
    if st.button("Show Train/Test Split"):
        c1,c2=st.columns(2)
        with c1: st.subheader("Train Head"); st.dataframe(train.head(10), use_container_width=True)
        with c2: st.subheader("Test Tail"); st.dataframe(test.tail(10), use_container_width=True)

    st.header("Stage 4 — Train Model")
    if st.button("Train"):
        m=Prophet(interval_width=conf/100, growth=growth, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(train)
        st.session_state["m"]=m; st.session_state["train"]=train; st.session_state["test"]=test
        st.success("Model trained.")

    st.header("Stage 5 — Forecast & Visualize")
    if "m" in st.session_state:
        m=st.session_state["m"]; train=st.session_state["train"]; test=st.session_state["test"]
        future=m.make_future_dataframe(periods=horizon_days, freq="D")
        forecast=m.predict(future)
        fig = px.line(forecast, x="ds", y="yhat", title=f"Forecast for {sku}")
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower", line=dict(dash="dot"))
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper", line=dict(dash="dot"))
        fig.add_scatter(x=train["ds"], y=train["y"], mode="markers", name="Train Actuals")
        if len(test)>0:
            fig.add_scatter(x=test["ds"], y=test["y"], mode="markers", name="Test Actuals")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["forecast"]=forecast
        with st.expander("Show Forecast Table"):
            st.dataframe(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon_days+5), use_container_width=True)
        if len(test)>0:
            joined = pd.merge(test, forecast[["ds","yhat"]], on="ds", how="left")
            joined["abs_err"] = (joined["y"] - joined["yhat"]).abs()
            mape = (joined["abs_err"] / joined["y"].replace(0, 1)).mean() * 100
            st.info(f"Simple Test MAPE over last {len(test)} days: **{mape:.2f}%** (lower is better).")
    else:
        st.info("Train first.")

    st.header("Stage 6 — Export")
    if "forecast" in st.session_state:
        out=st.session_state["forecast"][["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"Date","yhat":"Forecast","yhat_lower":"Lower","yhat_upper":"Upper"})
        csv=out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast CSV", data=csv, file_name=f"{sku}_forecast.csv", mime="text/csv")
        buffer=BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Forecast")
        st.download_button("Download Forecast Excel", data=buffer.getvalue(), file_name=f"{sku}_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Export available after forecasting.")        

if __name__ == "__main__":
    main()