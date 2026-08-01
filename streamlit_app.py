"""
Bangalore Care Desert Detector — Interactive Dashboard
========================================================
An ML-driven tool that flags healthcare "care deserts" among Bangalore's
243 BBMP wards. A Ridge regression model predicts how many pharmacies /
clinics / hospitals a ward "should" have given its geometric and spatial
profile (area, shape, distance from city center, local ward density).
Wards where the ACTUAL facility count falls far below the model's
prediction are flagged as ML-detected deserts — every flag comes with a
SHAP explanation of exactly which features drove the prediction.
"""

import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import shap
from shapely.geometry import Point
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Bangalore Care Desert Detector", layout="wide")

OUT = "outputs"


@st.cache_data
def load_data():
    gdf = gpd.read_file(f"{OUT}/ward_scores.geojson")
    with open(f"{OUT}/metrics.json") as f:
        metrics = json.load(f)
    X = pd.read_csv(f"{OUT}/model_features.csv")
    shap_values = np.load(f"{OUT}/shap_values.npy")
    return gdf, metrics, X, shap_values


@st.cache_resource
def load_model():
    model = joblib.load(f"{OUT}/model.joblib")
    scaler = joblib.load(f"{OUT}/scaler.joblib")
    return model, scaler


@st.cache_resource
def load_geolocator():
    return Nominatim(user_agent="bangalore_care_desert_detector_app")


gdf, metrics, X, shap_values = load_data()
model, scaler = load_model()
feature_cols = metrics["feature_cols"]

FEATURE_LABELS = {
    "log_area": "Ward area (log-scaled)",
    "perimeter_km": "Perimeter length",
    "compactness": "Shape compactness",
    "solidity": "Shape solidity",
    "dist_center_km": "Distance from city center",
    "local_ward_density_km": "Local ward density (avg dist to 5 nearest wards)",
    "num_neighbors": "Number of touching neighbor wards",
    "perimeter_area_ratio": "Perimeter-to-area ratio",
}

st.title("🏥 Bangalore Care Desert Detector")
st.caption(
    "An ML model predicts expected healthcare facility density per ward from its "
    "geometric/spatial profile. Wards where reality falls far short of the "
    "prediction are flagged as care deserts — with SHAP explaining why."
)

# ---------------------------------------------------------------------------
# Top-line model card
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wards analyzed", metrics["n_wards"])
c2.metric("Model", "Ridge Regression")
c3.metric("Cross-val R²", f"{metrics['cv_r2_mean']:.2f}")
c4.metric("Facilities mapped", f"{int(gdf['facility_count'].sum()):,}")

with st.expander("ℹ️ About the model & an honest look at its limits"):
    st.markdown(f"""
**Pipeline:** spatial join (facility points → wards) → 8 engineered geometric/spatial
features → Ridge regression (α=5.0, standardized features) → residual-based
anomaly scoring → SHAP explainability.

**Why Ridge, not a tree ensemble?** RandomForest and GradientBoosting were tested
via 5-fold cross-validation and actually generalized *worse* on this dataset
(CV R² ≈ 0.08–0.11 vs. **{metrics['cv_r2_mean']:.3f}** for Ridge, with higher variance
across folds). With only {metrics['n_wards']} wards and 8 features, a regularized
linear model is the better-fit choice — a real example of matching model
complexity to data size instead of defaulting to the fanciest option.

**Honest limitation:** ward *geometry* alone is a genuinely weak predictor of
facility count (best single-feature correlation ≈ 0.28–0.34). Facility placement
is driven mostly by commercial demand and population density, which this dataset
doesn't include. The R² of **{metrics['cv_r2_mean']:.2f}** reflects that — it's a
real, disclosed constraint, not a bug. The residual is still a meaningful signal:
it flags wards that are unusually under-served *even relative to what their shape
and location alone would predict*, which is a more defensible baseline than a
raw density formula.
""")

st.divider()

# ---------------------------------------------------------------------------
# NEW: "Find your area" — address / landmark search
# ---------------------------------------------------------------------------
st.subheader("📍 Find your area")
st.caption(
    "Type an address, landmark, or neighborhood name in Bangalore. We'll geocode it "
    "(free, via OpenStreetMap), find which BBMP ward it falls in, and show its "
    "care-desert score, rank among all 243 wards, and why the model scored it that way."
)

address_input = st.text_input(
    "e.g. 'Koramangala 5th Block', 'Indiranagar 100ft Road', 'MG Road, Bangalore'",
    key="address_search",
)

if address_input:
    with st.spinner("Looking that up..."):
        geolocator = load_geolocator()
        try:
            location = geolocator.geocode(f"{address_input}, Bangalore, Karnataka, India", timeout=10)
        except Exception:
            location = None

    if location is None:
        st.warning(
            "Couldn't find that location. Try being more specific (add a landmark, "
            "block number, or 'Bangalore') and search again."
        )
    else:
        pt = Point(location.longitude, location.latitude)
        match = gdf[gdf.geometry.contains(pt)]

        if match.empty:
            st.warning(
                f"Found **{location.address}**, but its coordinates fall outside the "
                "243 BBMP ward boundaries covered by this dataset — it may be just "
                "outside city limits."
            )
        else:
            found_idx = match.index[0]
            found_row = gdf.loc[found_idx]

            rank = int((gdf["desert_score"] > found_row["desert_score"]).sum()) + 1
            n_wards = len(gdf)
            percentile_underserved = 100 * (n_wards - rank + 1) / n_wards

            st.success(f"📍 **{location.address}** is in **{found_row['ward_name']}** ward")

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Actual facilities", int(found_row["facility_count"]))
            f2.metric("Model predicted", f"{found_row['predicted_facilities']:.1f}")
            f3.metric("Desert score", f"{found_row['desert_score']:.1f}")
            f4.metric("Under-served rank", f"#{rank} of {n_wards}")

            if found_row["desert_score"] > 5:
                st.info(
                    f"This ward ranks in the **top {percentile_underserved:.0f}%** most "
                    "under-served wards in the city — it has notably fewer healthcare "
                    "facilities than a ward with this shape, size, and location would "
                    "be expected to have. This is exactly the kind of ward a new "
                    "pharmacy or clinic would have the most impact in."
                )
            elif found_row["desert_score"] <= 0:
                st.info(
                    "This ward has as many (or more) facilities as the model predicted "
                    "— it is not flagged as under-served."
                )
            else:
                st.info(
                    f"This ward is mildly under-served relative to expectation "
                    f"(rank #{rank} of {n_wards})."
                )

            # Reuse the same SHAP explanation style as the ward drill-down below
            sv = shap_values[found_idx]
            contrib = pd.Series(
                sv, index=[FEATURE_LABELS.get(c, c) for c in feature_cols]
            ).sort_values()

            fig_search, ax_search = plt.subplots(figsize=(5, 3.2))
            colors_search = ["#d62728" if v < 0 else "#2ca02c" for v in contrib.values]
            ax_search.barh(contrib.index, contrib.values, color=colors_search)
            ax_search.axvline(0, color="black", linewidth=0.8)
            ax_search.set_xlabel("SHAP contribution to predicted facility count")
            ax_search.set_title(
                f"Why the model predicts {found_row['predicted_facilities']:.1f} "
                f"for {found_row['ward_name']}"
            )
            plt.tight_layout()
            st.pyplot(fig_search)

st.divider()

# ---------------------------------------------------------------------------
# Layout: map on the left, ranked list + ward drill-down on the right
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with right:
    st.subheader("🔎 Ranked care deserts")
    top_n = st.slider("Show top N most under-served wards", 5, 50, 15)
    ranked = gdf.sort_values("desert_score", ascending=False).head(top_n)
    display_df = ranked[["ward_name", "facility_count", "predicted_facilities", "desert_score"]].copy()
    display_df["predicted_facilities"] = display_df["predicted_facilities"].round(1)
    display_df["desert_score"] = display_df["desert_score"].round(1)
    display_df.columns = ["Ward", "Actual", "Predicted", "Desert score"]
    st.dataframe(display_df, use_container_width=True, height=350, hide_index=True)

    st.subheader("🧠 Explain a ward's prediction")
    ward_options = gdf.sort_values("desert_score", ascending=False)["ward_name"].tolist()
    selected_ward = st.selectbox("Pick a ward", ward_options)
    row_idx = gdf.index[gdf["ward_name"] == selected_ward][0]
    row = gdf.loc[row_idx]

    m1, m2, m3 = st.columns(3)
    m1.metric("Actual facilities", int(row["facility_count"]))
    m2.metric("Model predicted", f"{row['predicted_facilities']:.1f}")
    gap = row["predicted_facilities"] - row["facility_count"]
    m3.metric("Gap", f"{gap:.1f}", delta=None)

    sv = shap_values[row_idx]
    contrib = pd.Series(sv, index=[FEATURE_LABELS.get(c, c) for c in feature_cols])
    contrib = contrib.sort_values()

    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in contrib.values]
    ax.barh(contrib.index, contrib.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to predicted facility count")
    ax.set_title(f"Why the model predicts {row['predicted_facilities']:.1f} for {selected_ward}")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Green bars push the prediction UP (features that suggest this ward should "
        "have more facilities); red bars push it DOWN. This is what actually drove "
        "the number above, feature by feature."
    )

with left:
    st.subheader("🗺️ Care desert map")
    map_mode = st.radio(
        "Color wards by:",
        ["Desert score (ML-flagged gap)", "Actual facility count", "Model predicted count"],
        horizontal=False,
    )
    color_col = {
        "Desert score (ML-flagged gap)": "desert_score",
        "Actual facility count": "facility_count",
        "Model predicted count": "predicted_facilities",
    }[map_mode]

    center = [12.9716, 77.5946]
    fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf,
        columns=["ward_name", color_col],
        key_on="feature.properties.ward_name",
        fill_color="YlOrRd" if color_col != "predicted_facilities" else "YlGnBu",
        fill_opacity=0.75,
        line_opacity=0.3,
        legend_name=map_mode,
        nan_fill_color="lightgrey",
    ).add_to(fmap)

    tooltip = folium.GeoJsonTooltip(
        fields=["ward_name", "facility_count", "predicted_facilities", "desert_score"],
        aliases=["Ward:", "Actual facilities:", "Model predicted:", "Desert score:"],
        localize=True,
    )
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda x: {"fillOpacity": 0, "weight": 0},
        tooltip=tooltip,
    ).add_to(fmap)

    st_folium(fmap, width=None, height=560, returned_objects=[])

    st.caption(
        "Hover any ward for its actual facility count, the model's predicted count, "
        "and its desert score. Darker red = a bigger gap between what the model "
        "expected and what's actually there."
    )

st.divider()
st.subheader("📊 Global feature importance (SHAP)")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.Series(
    mean_abs_shap, index=[FEATURE_LABELS.get(c, c) for c in feature_cols]
).sort_values()

fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.barh(importance.index, importance.values, color="#4c72b0")
ax2.set_xlabel("Mean |SHAP value| — average impact on predicted facility count")
ax2.set_title("Which features drive the model's predictions overall?")
plt.tight_layout()
st.pyplot(fig2)

st.caption(
    "Built with GradientBoosting/RandomForest/Ridge compared via 5-fold CV, "
    "spatial joins in GeoPandas, and SHAP for explainability. "
    "Data: BBMP ward boundaries (DataMeet) + OpenStreetMap healthcare POIs."
)

