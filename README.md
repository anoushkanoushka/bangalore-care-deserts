# Bangalore Care Desert Detector

An ML tool that flags healthcare "care deserts" among Bangalore's 243 BBMP wards —
areas that have far fewer pharmacies, clinics, and hospitals than a model would
expect given their size, shape, and location in the city.

**[Live demo link — add after deploying to Streamlit Community Cloud]**

## The problem

Public health access mapping usually stops at "count facilities per area, rank by
density." That's a raw density formula, not a model — it can't tell you whether a
ward is *legitimately* underserved or just naturally smaller/more central. This
project instead asks: **given a ward's geometric and spatial profile, how many
facilities would we *expect* it to have — and where does reality fall furthest
short of that expectation?**

## Data

| Source | What | Size |
|---|---|---|
| [DataMeet Municipal Spatial Data](https://github.com/datameet/Municipal_Spatial_Data) | BBMP ward boundaries (2022 delimitation) | 243 wards |
| OpenStreetMap (Overpass API) | Pharmacy / clinic / hospital point locations | 2,755 facilities |

Both are free, open, community-maintained datasets — no API keys or paid tiers.

## Method

1. **Spatial join** — count how many facility points fall inside each ward polygon (GeoPandas `sjoin`).
2. **Feature engineering** — 8 geometric/spatial features per ward, computed in a projected metric CRS (UTM 43N):
   - log-scaled area, perimeter, compactness (isoperimetric quotient), solidity (vs. convex hull)
   - distance from city center, local ward density (mean distance to 5 nearest ward centroids — a proxy for how urbanized/tightly-packed that part of the city is)
   - number of directly touching neighbor wards, perimeter-to-area ratio
3. **Model selection via cross-validation** — Linear, Ridge, RandomForest, and GradientBoosting were compared with 5-fold CV. **Ridge regression (α=5.0) won**, generalizing better (CV R² ≈ 0.13) than the tree ensembles (R² ≈ 0.08–0.11), which overfit on only 243 samples. This is a deliberate example of matching model complexity to data size rather than defaulting to the fanciest option.
4. **Residual-based anomaly detection** — `desert_score = max(0, predicted − actual)`. Large positive scores mean a ward has far fewer facilities than its profile predicts.
5. **SHAP explainability** — every prediction (and therefore every flagged desert) can be broken down feature-by-feature, so "why is this ward flagged?" has a concrete answer, not just a number.

## Honest limitations

- **R² ≈ 0.13 is modest, and that's disclosed on purpose.** Ward geometry alone is a weak predictor of facility count (best single-feature correlation ≈ 0.28–0.34) because facility placement is driven mostly by commercial demand and population density — data this project doesn't have access to. The residual score is still meaningful (it flags wards under-served *relative to their own shape/location profile*), but it is not a claim of causal or complete explanation.
- OSM facility coverage is community-contributed and may be denser in some neighborhoods (tech-worker-mapped areas) than others — a ward showing 0 facilities could reflect a genuine gap or thinner OSM mapping in that pocket. Worth a manual spot-check (e.g. Google Maps search) before treating any single ward as ground truth.
- BBMP municipal wards ≠ census tracts, so this can't currently be combined with demographic vulnerability data (elderly %, income) without a separate spatial join — a natural next step.

## Findings (example)

Large peripheral wards such as **Anjanapura**, **Begur**, and **Rajarajeshwari
Nagar** have only 1–4 mapped facilities against a model expectation of 14–18 —
among the largest gaps in the city. By contrast, dense central wards like
**Hoysala Nagar** and **Nagapura** have 40–60+ facilities in under 2 sq km,
far above what their size alone would predict — a sign of genuine commercial/
medical clustering, not model error.

## Run it locally

```bash
pip install -r requirements.txt
python src/build_model.py      # regenerates outputs/ from raw data
streamlit run streamlit_app.py
```

Opens at `localhost:8501` with an interactive map (toggle between desert score /
actual / predicted facility counts), a ranked list of the most underserved
wards, and a per-ward SHAP explanation panel.

## Project structure

```
care_deserts_ml/
├── data/raw/                  # ward boundaries + OSM facility points
├── src/build_model.py         # full pipeline: join → features → model → SHAP → save
├── outputs/                   # ward_scores.csv/.geojson, model.joblib, shap_values.npy, metrics.json
├── streamlit_app.py           # interactive dashboard
├── requirements.txt
└── README.md
```

## Deploying the live demo

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the main file to `streamlit_app.py`.
3. Free hosting, live link in a few minutes — drop it at the top of this README.
