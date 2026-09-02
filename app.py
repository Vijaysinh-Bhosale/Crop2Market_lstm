import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import math
import requests
import pydeck as pdk
from pathlib import Path
from datetime import datetime, timedelta

from tensorflow.keras.models import load_model as keras_load_model

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from pincode_osrm import geocode_pincode
from streamlit_js_eval import get_geolocation

# ─────────────────────────────────────────────────────────────
#  HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────
def gmaps_nav_url(lat1, lon1, lat2, lon2):
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={lat1},{lon1}"
        f"&destination={lat2},{lon2}"
        "&travelmode=driving"
    )

def osrm_route_geojson(lat1, lon1, lat2, lon2):
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if resp.status_code != 200 or "routes" not in data or not data["routes"]:
        raise RuntimeError(f"OSRM error: {data}")
    route = data["routes"][0]
    return (
        route["geometry"]["coordinates"],
        route["distance"] / 1000.0,
        route["duration"] / 60.0,
    )

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
HIST_XLSX   = DATA_DIR / "Final_Cleaned_Dataset_After_Outlier_Removal.xlsx"
MODELS_DIR  = ROOT / "models_per_commodity" / "models"
SCALERS_DIR = ROOT / "models_per_commodity" / "scalers"
SEQ2SEQ_H   = 30
PLOT_H      = 7

@st.cache_resource(max_entries=8)
def load_commodity_model(name):
    key = name.replace(" ", "_")
    mp  = MODELS_DIR  / f"{key}_final.h5"
    fp  = SCALERS_DIR / f"{key}_feat.pkl"
    tp  = SCALERS_DIR / f"{key}_target.pkl"
    if not (mp.exists() and fp.exists() and tp.exists()):
        return None, None, None
    try:
        return (
            keras_load_model(str(mp), compile=False),
            joblib.load(str(fp)),
            joblib.load(str(tp)),
        )
    except Exception as e:
        st.error(f"Model load error ({name}): {e}")
        return None, None, None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@st.cache_data
def load_data_files():
    sd = pd.read_csv(DATA_DIR/"last_sequences.csv") if (DATA_DIR/"last_sequences.csv").exists() else None
    md = pd.read_csv(DATA_DIR/"markets.csv")        if (DATA_DIR/"markets.csv").exists()        else None
    meta = None
    mp = DATA_DIR / "last_sequences_metadata.json"
    if mp.exists():
        try: meta = json.loads(mp.read_text())
        except: pass
    return sd, md, meta

def infer_seq_features(meta, seqs):
    if meta:
        return int(meta["seq_len"]), list(meta["feature_order"])
    t0 = [c for c in seqs.columns if c.startswith("t0_")]
    fo = [c.replace("t0_","") for c in t0]
    tt = len([c for c in seqs.columns if c.startswith("t")])
    return tt // max(1, len(fo)), fo

def build_seq(row, seq_len, features):
    X = np.zeros((seq_len, len(features)), dtype=float)
    for t in range(seq_len):
        for j, f in enumerate(features):
            try: X[t,j] = float(row.get(f"t{t}_{f}", 0.0))
            except: pass
    return X

def predict_horizon(model, fs, ts, seq, horizon=SEQ2SEQ_H):
    sl, nf = seq.shape
    X  = seq.reshape(1, sl, nf)
    Xs = fs.transform(X.reshape(-1,nf)).reshape(1, sl, nf)
    ys = model.predict(Xs, verbose=0)
    return ts.inverse_transform(ys.reshape(-1,1)).reshape(horizon,).tolist()

def boosted_forecast(commodity, market, horizon=PLOT_H):
    if xgb is None or not HIST_XLSX.exists():
        return None, None
    try:
        raw = pd.read_excel(HIST_XLSX)
    except:
        return None, None
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ","_")
    if not {"commodity","market_name","price_date"}.issubset(raw.columns):
        return None, None
    sub = raw[(raw.commodity==commodity)&(raw.market_name==market)].copy()
    if sub.empty: return None, None
    sub["price_date"] = pd.to_datetime(sub["price_date"])
    sub = sub.sort_values("price_date")
    tc_cands = [c for c in sub.columns if "modal" in c and "price" in c]
    if not tc_cands: return None, None
    tc = tc_cands[0]
    for w in [7,14,30]: sub[f"roll{w}"] = sub[tc].rolling(w).mean()
    sub["diff1"]   = sub[tc].diff()
    sub["diff7"]   = sub[tc].diff(7)
    sub["day"]     = sub.price_date.dt.day
    sub["month"]   = sub.price_date.dt.month
    sub["weekday"] = sub.price_date.dt.weekday
    sub = sub.dropna()
    if len(sub) < 10: return None, None
    FEATS = [f for f in ["roll7","roll14","roll30","diff1","diff7","day","month","weekday"] if f in sub.columns]
    if len(FEATS) < 4: return None, None
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, verbosity=0)
    m.fit(sub[FEATS], sub[tc])
    last  = sub.iloc[-1].copy()
    vals  = sub[tc].tolist()
    preds, dates = [], []
    start = datetime.today().date() + timedelta(days=1)
    for i in range(horizon):
        nd = start + timedelta(days=i)
        dates.append(nd)
        feat = []
        for f in FEATS:
            if   f=="day":     feat.append(nd.day)
            elif f=="month":   feat.append(nd.month)
            elif f=="weekday": feat.append(nd.weekday())
            else:              feat.append(last.get(f, 0.0))
        p = float(m.predict(np.array([feat]))[0])
        preds.append(p)
        vals.append(p)
        last["roll30"]  = np.mean(vals[-30:])
        last["roll14"]  = np.mean(vals[-14:]) if len(vals)>=14 else last.get("roll14", last["roll30"])
        last["roll7"]   = np.mean(vals[-7:])  if len(vals)>=7  else last.get("roll7",  last["roll14"])
        last["diff1"]   = p - vals[-2] if len(vals)>=2 else last.get("diff1",0)
        last["diff7"]   = p - vals[-8] if len(vals)>=8 else last.get("diff7",last["diff1"])
        last["price_date"] = pd.Timestamp(nd)
    return pd.DataFrame({"price_date":dates,"Predicted Modal Price (₹)":preds}), float(np.mean(preds))

# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
seqs_df, markets_df, metadata = load_data_files()
if seqs_df is None or markets_df is None:
    st.error("Missing data files — place last_sequences.csv and markets.csv in data/")
    st.stop()

SEQ_LEN, FEATURE_NAMES = infer_seq_features(metadata, seqs_df)

ACCURATE_COORDS = {
    "Shree Chatrapati Shivaji Market Yard (Gultekdi, Pune)": (18.481876, 73.870052),
    "APMC Market (Vashi, Navi Mumbai)":                      (19.0744,   73.0112),
    "Solapur APMC Mandi (Solapur Market Yard)":              (17.680713, 75.927564),
    "Niphad Market (Nashik District)":                       (20.079964, 74.109314),
}
if "market_name" in markets_df.columns:
    markets_df["market_name"] = markets_df["market_name"].astype(str).str.strip()
    for mn,(la,lo) in ACCURATE_COORDS.items():
        mask = markets_df["market_name"]==mn
        if mask.any():
            markets_df.loc[mask,"latitude"]  = la
            markets_df.loc[mask,"longitude"] = lo

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop2Market", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS  — compacted for single-screen landing & cleaner cards
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #c9d1d9; }

/* wipe default block padding so the form fits one screen */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
div[data-testid="stHorizontalBlock"] { gap: 0.6rem !important; }

/* ── TOP BAR ── */
.topbar {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 10px 32px; display: flex; align-items: center; gap: 14px;
}
.topbar-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    border-radius: 9px; display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
}
.topbar-title { font-size: 1rem; font-weight: 800; color: #f0f6fc; margin: 0; letter-spacing: -.3px; }
.topbar-sub   { font-size: .68rem; color: #8b949e; margin: 0; }
.topbar-badge {
    margin-left: auto;
    background: rgba(35,134,54,.18); border: 1px solid rgba(35,134,54,.4);
    color: #3fb950; font-size: .64rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px; letter-spacing: .6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── LANDING — single compact card ─── */
.landing-wrap {
    padding: 18px 24px 14px 24px;
    max-width: 760px;
    margin: 0 auto;
}
.landing-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 18px 24px 20px 24px;
}
.landing-eyebrow {
    color: #3fb950; font-size: .62rem; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 4px;
}
.landing-title {
    font-size: 1.55rem; font-weight: 800; color: #f0f6fc;
    margin: 0 0 4px 0; letter-spacing: -.5px; line-height: 1.15;
}
.landing-desc {
    font-size: .82rem; color: #8b949e; margin: 0 0 12px 0; line-height: 1.4;
}
.form-section-label {
    font-size: .6rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: #6e7681;
    margin: 8px 0 3px 0; display: block;
}

/* CTA */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg,#238636,#1a7f37) !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: .9rem !important; border: none !important;
    border-radius: 9px !important; padding: 11px 0 !important;
    box-shadow: 0 4px 14px rgba(35,134,54,.32) !important;
    transition: opacity .15s !important; margin-top: 10px !important;
}
.stButton > button:hover { opacity: .9 !important; }

/* widget overrides — compact */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 7px !important;
    color: #c9d1d9 !important;
    font-size: .85rem !important;
    min-height: 34px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 2px rgba(31,111,235,.18) !important;
}
label, .stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label {
    font-size: .72rem !important; font-weight: 600 !important;
    color: #8b949e !important;
}
.stRadio > div { flex-direction: row !important; gap: 8px !important; }
.stRadio > div > label {
    background: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 7px !important; padding: 4px 14px !important;
    font-size: .78rem !important; font-weight: 600 !important;
    color: #c9d1d9 !important; cursor: pointer !important;
}
.stCheckbox > label {
    font-size: .78rem !important; color: #c9d1d9 !important; font-weight: 500 !important;
}
.stDownloadButton > button {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
    font-size: .8rem !important; font-weight: 600 !important; padding: 7px 18px !important;
}

/* ─── RESULTS PAGE ─── */
.results-wrap { padding: 16px 28px 32px 28px; }

.summary-bar {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 10px 18px; display: flex; align-items: center; gap: 16px;
    margin-bottom: 12px; flex-wrap: wrap;
}
.summary-bar-tag { font-size: .58rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #8b949e; margin-bottom: 2px; }
.summary-bar-val { font-size: .82rem; font-weight: 700; color: #f0f6fc; }
.summary-sep { width: 1px; height: 24px; background: #30363d; flex-shrink: 0; }

.kpi-row { display: flex; gap: 8px; margin-bottom: 12px; }
.kpi-tile { flex: 1; background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px 14px; }
.kpi-label { font-size: .56rem; font-weight: 700; letter-spacing: .9px; text-transform: uppercase; color: #8b949e; margin-bottom: 3px; }
.kpi-value { font-size: 1.15rem; font-weight: 800; color: #f0f6fc; font-family: 'JetBrains Mono', monospace; letter-spacing: -.5px; }
.kpi-value.green { color: #3fb950; }
.kpi-value.blue  { color: #58a6ff; }
.kpi-value.amber { color: #d29922; }

.rec-card {
    background: linear-gradient(135deg, rgba(35,134,54,.08), rgba(31,111,235,.06));
    border: 1px solid rgba(35,134,54,.3); border-radius: 13px;
    padding: 16px 22px; margin-bottom: 14px;
    display: flex; align-items: flex-start; gap: 22px; flex-wrap: wrap;
}
.rec-left { flex: 1; min-width: 240px; }
.rec-badge { font-size: .6rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #3fb950; margin-bottom: 4px; }
.rec-market { font-size: 1.4rem; font-weight: 800; color: #f0f6fc; letter-spacing: -.4px; margin-bottom: 4px; line-height: 1.2; }
.rec-net { font-size: 1.7rem; font-weight: 800; color: #3fb950; font-family: 'JetBrains Mono', monospace; letter-spacing: -1px; line-height: 1; margin: 4px 0; }
.rec-meta { font-size: .73rem; color: #8b949e; line-height: 1.6; }
.rec-meta span { color: #c9d1d9; font-weight: 600; }
.nav-btn a {
    display: inline-flex; align-items: center; gap: 7px;
    background: #1f6feb; color: #fff !important;
    font-weight: 700; font-size: .8rem;
    padding: 8px 16px; border-radius: 8px; text-decoration: none; white-space: nowrap;
}
.nav-btn a:hover { opacity: .85; }

.sec-heading {
    font-size: .78rem; font-weight: 700; color: #f0f6fc;
    margin: 10px 0 8px 0; display: flex; align-items: center; gap: 10px;
}
.sec-heading::after { content:''; flex:1; height:1px; background:#21262d; margin-left:6px; }

.map-caption { font-size: .7rem; color: #8b949e; margin-top: 4px; }
.map-caption strong { color: #c9d1d9; }

.stDataFrame { border: 1px solid #30363d !important; border-radius: 10px !important; overflow: hidden !important; }

.footer-note { font-size: .64rem; color: #484f58; font-family: 'JetBrains Mono', monospace; margin-top: 20px; padding-top: 10px; border-top: 1px solid #21262d; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
for k, v in {
    "page": "landing",
    "gps_lat": None, "gps_lon": None,
    "results": None,
    "inputs": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-icon">🌾</div>
  <div>
    <div class="topbar-title">Crop2Market</div>
    <div class="topbar-sub">Market Strategy Advisor — LSTM + XGBoost hybrid forecasting</div>
  </div>
  <div class="topbar-badge">LIVE FORECAST</div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
#  PAGE 1 — LANDING / INPUT FORM  (single card, fits one screen)
# ═════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    # Single outer wrap centred + width-limited via CSS
    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)

    # ONE card with header inside — no second empty rectangle
    with st.container(border=True):
        st.markdown("""
        <div class="landing-eyebrow">Maharashtra Agricultural Markets</div>
        <div class="landing-title">Find your best market today</div>
        <div class="landing-desc">Enter your crop, quantity, and location. We'll rank nearby mandis by net revenue using live price forecasts.</div>
        """, unsafe_allow_html=True)

        # Two-column layout: left = Commodity + Location, right = Unit + Quantity + Transport
        left_col, right_col = st.columns([1.3, 1], gap="large")

        with left_col:
            st.markdown('<span class="form-section-label">Commodity</span>', unsafe_allow_html=True)
            commodity = st.selectbox(
                "Commodity",
                sorted(seqs_df["commodity"].unique()),
                index=0, label_visibility="collapsed", key="inp_commodity",
            )

            st.markdown('<span class="form-section-label">Your Location</span>', unsafe_allow_html=True)
            use_gps = st.checkbox("📍 Use my current GPS location", key="inp_gps")
            if not use_gps:
                pincode = st.text_input("Pincode", placeholder="e.g. 411001",
                                        label_visibility="collapsed", key="inp_pincode")
            else:
                pincode = ""
                try:
                    loc = get_geolocation()
                    if loc and loc.get("coords"):
                        st.session_state.gps_lat = loc["coords"]["latitude"]
                        st.session_state.gps_lon = loc["coords"]["longitude"]
                        st.success(f"GPS ready — ({st.session_state.gps_lat:.4f}, {st.session_state.gps_lon:.4f})")
                except Exception as e:
                    st.warning(f"GPS error: {e}")

        with right_col:
            st.markdown('<span class="form-section-label">Unit</span>', unsafe_allow_html=True)
            unit = st.radio("Unit", ("Quintals", "Kilograms"),
                            horizontal=True, label_visibility="collapsed", key="inp_unit")

            st.markdown('<span class="form-section-label">Quantity</span>', unsafe_allow_html=True)
            qty = st.number_input("Amount", min_value=0.1, value=50.0, step=1.0,
                                  label_visibility="collapsed", key="inp_qty")

            st.markdown('<span class="form-section-label">Transport ₹/t/km</span>', unsafe_allow_html=True)
            transport_rate = st.number_input(
                "Transport", min_value=0.1, value=5.0, step=0.5,
                label_visibility="collapsed", key="inp_transport",
            )

        go_btn = st.button("🔍  Find Best Markets")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── On button press: compute & switch page ──
    if go_btn:
        qty_quintals = (qty / 100.0) if unit == "Kilograms" else qty
        qty_tonnes   = (qty_quintals * 100.0) / 1000.0

        user_lat = user_lon = None
        if use_gps:
            if st.session_state.gps_lat and st.session_state.gps_lon:
                user_lat, user_lon = st.session_state.gps_lat, st.session_state.gps_lon
            else:
                st.warning("GPS location not yet available. Allow access and try again.")
                st.stop()
        elif pincode:
            try:
                user_lat, user_lon = geocode_pincode(pincode)
            except Exception as e:
                st.warning(f"Could not geocode pincode ({e}). Using fallback.")
        if user_lat is None:
            user_lat = float(markets_df["latitude"].mean())
            user_lon = float(markets_df["longitude"].mean())

        model_c, feat_s, targ_s = load_commodity_model(commodity)
        seqs_comm = seqs_df[seqs_df["commodity"] == commodity]

        feat_names = FEATURE_NAMES
        if feat_s is not None and hasattr(feat_s, "feature_names_in_"):
            try: feat_names = list(feat_s.feature_names_in_)
            except: pass
        elif metadata and metadata.get("feature_order"):
            feat_names = list(metadata["feature_order"])

        if feat_s is not None and hasattr(feat_s, "n_features_in_"):
            en = int(feat_s.n_features_in_)
            if len(feat_names) != en:
                if hasattr(feat_s, "feature_names_in_"):
                    feat_names = list(feat_s.feature_names_in_)
                else:
                    t0c = [c for c in seqs_df.columns if c.startswith("t0_")]
                    feat_names = [c.replace("t0_","") for c in t0c][:en]

        rows, fcasts, paths = [], {}, {}
        with st.spinner("Calculating routes and forecasts…"):
            for _, mr in markets_df.iterrows():
                mn = mr["market_name"]
                try:
                    mlat, mlon = float(mr["latitude"]), float(mr["longitude"])
                except:
                    continue

                path_c = None
                try:
                    path_c, dist, dur = osrm_route_geojson(user_lat, user_lon, mlat, mlon)
                except:
                    dist = haversine_km(user_lat, user_lon, mlat, mlon)
                    dur  = float("nan")
                paths[mn] = path_c

                sr = seqs_comm[seqs_comm["market_name"]==mn]
                if sr.empty: sr = seqs_df[seqs_df["market_name"]==mn]
                if sr.empty: continue

                seq = build_seq(sr.iloc[0].to_dict(), SEQ_LEN, feat_names)

                if model_c is not None:
                    try:
                        y30 = predict_horizon(model_c, feat_s, targ_s, seq)
                        pp  = float(np.nanmean(y30))
                        fcasts[mn] = y30
                    except:
                        pp = float(seq[-1,0]); fcasts[mn] = [pp]*SEQ2SEQ_H
                else:
                    pp = float(seq[-1,0]); fcasts[mn] = [pp]*SEQ2SEQ_H

                bdf, _ = boosted_forecast(commodity, mn)
                if bdf is not None:
                    fcasts[mn] = bdf["Predicted Modal Price (₹)"].tolist()

                rev  = pp * qty_quintals
                tc_  = transport_rate * qty_tonnes * dist
                rows.append({
                    "market_name":    mn,
                    "latitude":       mlat,
                    "longitude":      mlon,
                    "distance_km":    round(dist, 2),
                    "travel_time_min":dur,
                    "pred_price":     round(pp, 2),
                    "revenue":        round(rev, 2),
                    "travel_cost":    round(tc_, 2),
                    "net_revenue":    round(rev - tc_, 2),
                })

        if not rows:
            st.error("No market data found for this commodity.")
            st.stop()

        res = pd.DataFrame(rows)
        for c in ["pred_price","distance_km","travel_time_min","travel_cost","revenue","net_revenue"]:
            res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)
        res = res.sort_values("net_revenue", ascending=False).reset_index(drop=True)

        st.session_state.results = {
            "res_df":      res,
            "forecasts":   fcasts,
            "route_paths": paths,
            "user_lat":    user_lat,
            "user_lon":    user_lon,
        }
        st.session_state.inputs = {
            "commodity":      commodity,
            "qty":            qty,
            "unit":           unit,
            "pincode":        pincode,
            "transport_rate": transport_rate,
        }
        st.session_state.page = "results"
        st.rerun()


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — RESULTS  (map + ranked table side-by-side, smaller map)
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    R   = st.session_state.results
    INP = st.session_state.inputs

    res_df    = R["res_df"]
    fcasts    = R["forecasts"]
    paths     = R["route_paths"]
    user_lat  = R["user_lat"]
    user_lon  = R["user_lon"]
    commodity = INP["commodity"]

    best      = res_df.iloc[0]
    best_lat  = float(best["latitude"])
    best_lon  = float(best["longitude"])
    nav_url   = gmaps_nav_url(user_lat, user_lon, best_lat, best_lon)
    best_eta  = best["travel_time_min"]
    eta_text  = f"{best_eta:.0f} min" if (not isinstance(best_eta, float) or not math.isnan(best_eta)) and best_eta > 0 else "N/A"
    best_path = paths.get(best["market_name"])

    st.markdown('<div class="results-wrap">', unsafe_allow_html=True)

    # Summary bar
    st.markdown(f"""
    <div class="summary-bar">
      <div><div class="summary-bar-tag">Commodity</div><div class="summary-bar-val">{INP['commodity']}</div></div>
      <div class="summary-sep"></div>
      <div><div class="summary-bar-tag">Quantity</div><div class="summary-bar-val">{INP['qty']} {INP['unit']}</div></div>
      <div class="summary-sep"></div>
      <div><div class="summary-bar-tag">Location</div><div class="summary-bar-val">{INP['pincode'] if INP['pincode'] else 'GPS'}</div></div>
      <div class="summary-sep"></div>
      <div><div class="summary-bar-tag">Transport</div><div class="summary-bar-val">₹{INP['transport_rate']}/t/km</div></div>
    </div>
    """, unsafe_allow_html=True)

    edit_col, _ = st.columns([1, 5])
    with edit_col:
        if st.button("← Edit Inputs"):
            st.session_state.page = "landing"
            st.rerun()

    # KPI tiles
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-tile"><div class="kpi-label">Est. Net Revenue</div><div class="kpi-value green">₹{float(best['net_revenue']):,.0f}</div></div>
      <div class="kpi-tile"><div class="kpi-label">Predicted Price</div><div class="kpi-value">₹{float(best['pred_price']):,.0f}/q</div></div>
      <div class="kpi-tile"><div class="kpi-label">Distance</div><div class="kpi-value blue">{float(best['distance_km']):.1f} km</div></div>
      <div class="kpi-tile"><div class="kpi-label">Drive Time</div><div class="kpi-value amber">{eta_text}</div></div>
      <div class="kpi-tile"><div class="kpi-label">Travel Cost</div><div class="kpi-value">₹{float(best['travel_cost']):,.0f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Top recommendation
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-left">
        <div class="rec-badge">🏆 Top Recommendation</div>
        <div class="rec-market">{best['market_name']}</div>
        <div class="rec-net">₹ {float(best['net_revenue']):,.2f}</div>
        <div class="rec-meta">
          Predicted price <span>₹{float(best['pred_price']):.2f}/quintal</span> ·
          Travel cost <span>₹{float(best['travel_cost']):.2f}</span> ·
          Distance <span>{float(best['distance_km']):.2f} km</span> ·
          ETA <span>{eta_text}</span>
        </div>
      </div>
      <div class="nav-btn"><a href="{nav_url}" target="_blank">🗺️  Navigate in Google Maps</a></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Route Map + All Markets Ranked SIDE BY SIDE ──
    map_col, table_col = st.columns([1, 1], gap="medium")

    with map_col:
        st.markdown('<div class="sec-heading">Route Map</div>', unsafe_allow_html=True)
        if best_path:
            mid_lat = (user_lat + best_lat) / 2
            mid_lon = (user_lon + best_lon) / 2
            dist_deg = haversine_km(user_lat, user_lon, best_lat, best_lon)
            zoom = 10 if dist_deg < 30 else (8 if dist_deg < 100 else 6)

            deck = pdk.Deck(
                layers=[
                    pdk.Layer("PathLayer",
                        data=[{"path": best_path}], get_path="path",
                        get_color=[250, 140, 20],
                        width_min_pixels=4, width_max_pixels=8),
                    pdk.Layer("ScatterplotLayer",
                        data=[{"lat": user_lat, "lon": user_lon, "label": "You"}],
                        get_position="[lon, lat]",
                        get_fill_color=[239, 68, 68], get_radius=600, pickable=True),
                    pdk.Layer("ScatterplotLayer",
                        data=[{"lat": best_lat, "lon": best_lon, "label": best["market_name"]}],
                        get_position="[lon, lat]",
                        get_fill_color=[35, 134, 54], get_radius=600, pickable=True),
                ],
                initial_view_state=pdk.ViewState(
                    latitude=mid_lat, longitude=mid_lon, zoom=zoom, pitch=0,
                ),
                map_style="road",
                tooltip={"text": "{label}"},
                map_provider="carto",
                parameters={},
                views=[pdk.View(type="MapView", controller=True)],
            )
            # ↓↓ Halved height (was 380) ↓↓
            st.pydeck_chart(deck, height=320)
            st.markdown(
                f'<div class="map-caption">🔴 Your location · 🟢 <strong>{best["market_name"]}</strong> · '
                f'{float(best["distance_km"]):.2f} km · ETA {eta_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Road route unavailable — straight-line distance used.")

    with table_col:
        st.markdown('<div class="sec-heading">All Markets Ranked</div>', unsafe_allow_html=True)

        table_df = (
            res_df[[
                "market_name","pred_price","distance_km",
                "travel_time_min","travel_cost","revenue","net_revenue",
            ]]
            .rename(columns={
                "market_name":    "Market",
                "pred_price":     "Price (₹/q)",
                "distance_km":    "Dist (km)",
                "travel_time_min":"ETA (min)",
                "travel_cost":    "Cost (₹)",
                "revenue":        "Revenue (₹)",
                "net_revenue":    "Net Rev (₹)",
            })
        )

        net_vals = table_df["Net Rev (₹)"]
        nmin, nmax = net_vals.min(), net_vals.max()
        nrng = max(nmax - nmin, 1)
        def _style_net(v):
            intensity = (v - nmin) / nrng
            g = int(80 + intensity * 105)
            return f"color: rgb(63,{g},80); font-weight: 700"

        st.dataframe(
            table_df.style
                .format({
                    "Price (₹/q)": "{:.2f}",
                    "Dist (km)":   "{:.2f}",
                    "ETA (min)":   "{:.1f}",
                    "Cost (₹)":    "{:.2f}",
                    "Revenue (₹)": "{:.2f}",
                    "Net Rev (₹)": "{:.2f}",
                })
                .map(_style_net, subset=["Net Rev (₹)"]),
            height=320,
            width="stretch",
            hide_index=True,
        )

    # ── 7-Day Forecast ──
    st.markdown(
        f'<div class="sec-heading">7-Day Price Forecast — {best["market_name"]}</div>',
        unsafe_allow_html=True,
    )

    fcast_df, _ = boosted_forecast(commodity, best["market_name"])
    if fcast_df is None:
        sv = fcasts.get(best["market_name"], [float(best["pred_price"])]*PLOT_H)[:PLOT_H]
        fcast_df = pd.DataFrame({
            "price_date": [datetime.today().date()+timedelta(days=i+1) for i in range(PLOT_H)],
            "Predicted Modal Price (₹)": sv,
        })
    fcast_df["price_date"] = pd.to_datetime(fcast_df["price_date"])

    if go is None:
        st.line_chart(fcast_df.set_index("price_date")["Predicted Modal Price (₹)"])
    else:
        prices = fcast_df["Predicted Modal Price (₹)"].tolist()
        avg_p  = np.mean(prices)
        colors = ["#3fb950" if p >= avg_p else "#58a6ff" for p in prices]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fcast_df["price_date"], y=prices,
            fill="tozeroy", mode="none", showlegend=False,
            hoverinfo="skip", fillcolor="rgba(88,166,255,.06)",
        ))
        fig.add_trace(go.Scatter(
            x=fcast_df["price_date"], y=prices,
            mode="lines+markers",
            line=dict(width=2.5, shape="spline", smoothing=1.1, color="#58a6ff"),
            marker=dict(size=8, color=colors, line=dict(width=1.5, color="#0d1117")),
            hovertemplate="%{x|%A, %b %d}<br>₹ %{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(
            y=avg_p, line_dash="dot", line_color="rgba(139,148,158,.4)",
            annotation_text=f"avg ₹{avg_p:,.0f}",
            annotation_font_color="#8b949e", annotation_font_size=11,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=20, t=20, b=10),
            height=240,
            xaxis=dict(tickformat="%b %d", showgrid=False,
                       tickfont=dict(size=11,color="#8b949e"), linecolor="#30363d"),
            yaxis=dict(showgrid=True, gridcolor="rgba(48,54,61,.6)",
                       tickprefix="₹ ", tickfont=dict(size=11,color="#8b949e"), zeroline=False),
            showlegend=False,
            hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d",
                            font_size=12, font_color="#c9d1d9"),
        )
        st.plotly_chart(fig, width="stretch", config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d","lasso2d","toggleSpikelines","autoScale2d"],
            "responsive": True,
        })

    # ── Download ──
    st.download_button(
        "⬇  Download Results as CSV",
        res_df.drop(columns=["latitude","longitude"], errors="ignore")
              .to_csv(index=False).encode("utf-8"),
        file_name=f"market_recommendations_{commodity.replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="footer-note">Powered by LSTM Seq2Seq + XGBoost hybrid · Route data via OSRM</div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)
