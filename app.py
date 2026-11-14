import calendar
import os

import duckdb
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

st.set_page_config(page_title="India Burning Activity Calendar", page_icon="🔥", layout="wide")
pio.templates.default = "plotly_dark"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
h1 {font-weight: 800; letter-spacing: -0.5px;}
div[data-testid="stMetric"] {
    background: #1a1f2b; border: 1px solid #2a3142;
    border-radius: 12px; padding: 14px 18px;}
div[data-testid="stMetricValue"] {color: #ff6b35;}
section[data-testid="stSidebar"] {border-right: 1px solid #2a3142;}
</style>
""", unsafe_allow_html=True)

GRID = "Data/harmonized_monthly.parquet"
FIRST = pd.Timestamp("2000-11-01")
AQUA_START = pd.Timestamp("2002-08-01")   # before this: Terra only, scaled
LAST_MONTH = pd.Timestamp("2026-11-01")   # partial month (data ends 2 Nov 2026)
MONTHS = list(calendar.month_abbr)[1:]
ACCENT = "#ff6b35"
TRANSPARENT = "rgba(0,0,0,0)"

PRESETS = {
    "All India": (68.0, 6.5, 97.5, 37.5),
    "Punjab & Haryana (crop burning)": (73.8, 27.6, 77.6, 32.6),
    "Rajasthan": (69.4, 23.0, 78.3, 30.2),
    "Odisha": (81.3, 17.8, 87.5, 22.6),
    "Uttarakhand (forest fires)": (77.5, 28.7, 81.1, 31.5),
    "North-East India": (89.7, 21.9, 97.4, 29.5),
}


@st.cache_data
def load_grid():
    g = pd.read_parquet(GRID)
    g["month"] = pd.to_datetime(g["month"])
    g["cx"] = g["lon0"] + 0.125
    g["cy"] = g["lat0"] + 0.125
    return g[["month", "cx", "cy", "n_harm"]]


def aoi_series(g, w, s, e, n):
    sel = g[(g.cx >= w) & (g.cx <= e) & (g.cy >= s) & (g.cy <= n)]
    ser = sel.groupby("month")["n_harm"].sum()
    ser = ser.reindex(pd.date_range(FIRST, LAST_MONTH, freq="MS"), fill_value=0.0)
    df = ser.rename("n").rename_axis("month").reset_index()
    df["year"] = df.month.dt.year
    df["cal"] = df.month.dt.month
    df["estimated"] = df.month < AQUA_START
    df["partial"] = df.month == LAST_MONTH
    return df


def make_map(bbox, draw):
    w, s, e, n = bbox
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=4, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Dark",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    m.fit_bounds([[s, w], [n, e]])
    if draw:
        Draw(
            draw_options={"rectangle": True, "polygon": True, "polyline": False,
                          "circle": False, "marker": False, "circlemarker": False},
            edit_options={"edit": False},
        ).add_to(m)
    else:
        folium.Rectangle([[s, w], [n, e]], color=ACCENT, weight=2,
                         fill=True, fill_opacity=0.12).add_to(m)
    return m


def pivot(view, col):
    return view.pivot(index="year", columns="cal", values=col).reindex(columns=range(1, 13))


def heatmap(view, mode):
    anomaly = mode.startswith("Anomaly")
    col = "anom_pct" if anomaly else "n"
    raw = pivot(view, col)
    est = pivot(view.assign(e=view.estimated.astype(float)), "e")
    stars = np.where(est.values == 1, "*", "")
    z = raw.clip(-150, 150) if anomaly else raw
    fig = go.Figure(go.Heatmap(
        z=z.values, x=MONTHS, y=raw.index, text=stars, texttemplate="%{text}",
        customdata=raw.values,
        hovertemplate=("%{y} %{x}<br>" + ("%{customdata:.0f}% vs median"
                       if anomaly else "%{customdata:,.0f} hotspots") + "<extra></extra>"),
        colorscale="RdBu_r" if anomaly else "Viridis",
        zmid=0 if anomaly else None,
        colorbar=dict(title="% vs median" if anomaly else "Hotspots"),
        xgap=1, ygap=1,
    ))
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_xaxes(side="top")
    fig.update_layout(height=max(380, 22 * len(raw) + 120),
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT)
    return fig


def seasonal_chart(view):
    full = view[~view.partial]
    q = full.groupby("cal")["n"].quantile([0.1, 0.5, 0.9]).unstack()
    fig = go.Figure()
    fig.add_scatter(x=MONTHS, y=q[0.9], mode="lines", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=MONTHS, y=q[0.1], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(200,200,200,0.20)", name="10th-90th percentile")
    fig.add_scatter(x=MONTHS, y=q[0.5], mode="lines", line=dict(color="#e6e6e6", width=2),
                    name="Median")
    cur_year = LAST_MONTH.year
    cur = full[full.year == cur_year].set_index("cal")["n"].reindex(range(1, 13))
    if cur.notna().any():
        fig.add_scatter(x=MONTHS, y=cur.values, mode="lines+markers",
                        line=dict(color=ACCENT, width=3), name=str(cur_year))
    fig.update_layout(height=360, paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                      yaxis_title="Hotspots (MODIS-equivalent)",
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
    return fig


def hotspot_query(w, s, e, n, d0, d1, sensors):
    parts = [
        f"""SELECT latitude, longitude, CAST(acq_date AS DATE) AS acq_date, acq_time,
                   sensor, satellite, CAST(confidence AS VARCHAR) AS confidence, frp, daynight
            FROM 'Data/clean_{x}.parquet'
            WHERE latitude BETWEEN {s} AND {n} AND longitude BETWEEN {w} AND {e}
              AND acq_date BETWEEN DATE '{d0}' AND DATE '{d1}'"""
        for x in sensors
    ]
    return " UNION ALL ".join(parts)


# ---------------------------------------------------------------- header
st.title("🔥 India Burning Activity Calendar")
st.caption("Harmonized MODIS + VIIRS active-fire hotspots · NASA FIRMS · Nov 2000 to Nov 2026 · "
           "Select an area in the sidebar to compare any month against its historical median")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Area of interest")
    method = st.radio("Select area by", ["Preset region", "Type coordinates", "Draw on map"])
    bbox = None
    if method == "Preset region":
        bbox = PRESETS[st.selectbox("Region", list(PRESETS))]
    elif method == "Type coordinates":
        w = st.number_input("West (lon)", value=68.0, format="%.3f")
        s = st.number_input("South (lat)", value=6.5, format="%.3f")
        e = st.number_input("East (lon)", value=97.5, format="%.3f")
        n = st.number_input("North (lat)", value=37.5, format="%.3f")
        bbox = (w, s, e, n)
    y0, y1 = st.slider("Years", 2000, 2026, (2000, 2026))
    mode = st.radio("Calendar shows", ["Hotspot counts", "Anomaly vs median (%)"])

# ---------------------------------------------------------------- map
if method == "Draw on map":
    st.markdown("Draw a rectangle or polygon on the map (toolbar at left).")
    out = st_folium(make_map(PRESETS["All India"], True), height=420, width=None,
                    key="aoi_map", returned_objects=["last_active_drawing"])
    drawing = (out or {}).get("last_active_drawing")
    if drawing:
        pts = drawing["geometry"]["coordinates"][0]
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        bbox = (min(lons), min(lats), max(lons), max(lats))
    else:
        st.info("Nothing drawn yet, so showing all of India.")
        bbox = PRESETS["All India"]
else:
    st_folium(make_map(bbox, False), height=320, width=None, key="view_map",
              returned_objects=[])

w, s, e, n = bbox
if w >= e or s >= n:
    st.error("West must be less than East, and South less than North.")
    st.stop()
st.write(f"**Selected box:** W {w:.2f}, S {s:.2f}, E {e:.2f}, N {n:.2f}")

# ---------------------------------------------------------------- analysis
g = load_grid()
df = aoi_series(g, w, s, e, n)
view = df[(df.year >= y0) & (df.year <= y1)].copy()
if view.n.sum() == 0:
    st.warning("No hotspots found in this area and period.")
    st.stop()

ref = view[~view.partial].groupby("cal")["n"].median()
view["median"] = view.cal.map(ref)
view["anom_pct"] = np.where(view["median"] > 0,
                            (view.n - view["median"]) / view["median"] * 100, np.nan)
view.loc[view.partial, ["n", "anom_pct"]] = np.nan

# headline numbers for the latest full month in the range
last = view[~view.partial].iloc[-1]
same = view[(~view.partial) & (view.cal == last.cal)]
rank = int((same.n > last.n).sum()) + 1
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{last.month:%b %Y} hotspots", f"{last.n:,.0f}")
c2.metric(f"Median for {MONTHS[int(last.cal) - 1]}", f"{last['median']:,.0f}")
c3.metric("Anomaly vs median",
          "n/a" if np.isnan(last.anom_pct) else f"{last.anom_pct:+.0f}%")
c4.metric("Rank among same months", f"{rank} of {len(same)}")

st.subheader("Burning activity calendar")
st.plotly_chart(heatmap(view, mode), width="stretch")
st.caption("* = estimated from Terra only (before Aug 2002), lower certainty. "
           "Blank = no data or incomplete month (Nov 2026 has only 2 days).")

left, right = st.columns(2)
with left:
    st.subheader("Seasonal cycle")
    st.plotly_chart(seasonal_chart(view), width="stretch")
with right:
    st.subheader("Top 10 months")
    top = view[~view.partial].nlargest(10, "n").copy()
    top["Month"] = top.month.dt.strftime("%b %Y")
    top["Hotspots"] = top.n.round(0).astype(int)
    top["Note"] = np.where(top.estimated, "estimated (Terra only)", "")
    st.dataframe(top[["Month", "Hotspots", "Note"]], hide_index=True, width="stretch")

# ---------------------------------------------------------------- downloads
st.subheader("Download")
d1c, d2c = st.columns(2)
with d1c:
    outdf = view.rename(columns={"n": "hotspots_harmonized", "median": "median_same_month",
                                 "anom_pct": "anomaly_pct"})
    outdf["month"] = outdf.month.dt.strftime("%Y-%m")
    st.download_button("⬇ Monthly harmonized counts (CSV)",
                       outdf[["month", "hotspots_harmonized", "median_same_month",
                              "anomaly_pct", "estimated", "partial"]].to_csv(index=False),
                       file_name="monthly_hotspots.csv", mime="text/csv")
with d2c:
    HAS_HS = os.path.exists("Data/clean_modis.parquet")
    if not HAS_HS:
        st.info("Hotspot-level CSV is available in the local version only. "
                "The monthly CSV on the left works everywhere.")
    sensors = st.multiselect("Sensors for hotspot-level CSV",
                             ["modis", "snpp", "n20", "n21"],
                             default=["modis", "snpp"]) if HAS_HS else []
    sig = (round(w, 3), round(s, 3), round(e, 3), round(n, 3), y0, y1, tuple(sensors))
    if HAS_HS and st.button("Prepare hotspot-level CSV") and sensors:
        q = hotspot_query(w, s, e, n, f"{y0}-01-01", f"{y1}-12-31", sensors)
        con = duckdb.connect()
        count = con.execute(f"SELECT count(*) FROM ({q})").fetchone()[0]
        if count > 2_000_000:
            st.warning(f"{count:,} rows is too many. Select a smaller area or fewer years.")
        else:
            res = con.execute(q + " ORDER BY acq_date, acq_time").df()
            st.session_state["hs"] = (sig, res.to_csv(index=False), count)
    if HAS_HS and "hs" in st.session_state and st.session_state["hs"][0] == sig:
        st.download_button(f"⬇ {st.session_state['hs'][2]:,} hotspots (CSV)",
                           st.session_state["hs"][1], file_name="hotspots_aoi.csv",
                           mime="text/csv")
        st.caption("Hotspot-level rows are raw detections per sensor, not scaled.")

with st.expander("Method and limitations"):
    st.markdown("""
- **Source:** NASA FIRMS MODIS C6.1 and VIIRS (S-NPP) active fires over India. Low-confidence
  detections and non-vegetation sources (volcano, industry, offshore) are removed.
- **Harmonization:** from Feb 2012 VIIRS counts are divided by a calendar-month ratio to
  MODIS (calibrated on the 2012-2026 overlap). Before Aug 2002 only Terra flew, so counts are
  scaled up by a calendar-month Terra to Terra+Aqua factor.
- **Accuracy:** on held-out months (2021-2026) the median monthly error of the VIIRS scaling
  was about 12%; 90% of months were within about +/-34%.
- **Limits:** scaling factors are national, so small areas may be less accurate.
  Overpass times differ between satellites and drift over time. Counts are hotspots,
  not burned area.
""")