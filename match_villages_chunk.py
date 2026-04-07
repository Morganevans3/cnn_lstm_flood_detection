'''
Created by Morgan

step 4: Match daily climate features to village coordinates (nearest neighbor with k = 3, 5, 10).
- Input:
    ../../../Data/CDS_Climate/Processed/daily_features/daily_features.csv
    ../../../../Pakistan_Villages/village_coordinates.csv
- Output:
    ../../../Data/CDS_Climate/Processed/panels/daily_panel_k3.csv
    ../../../Data/CDS_Climate/Processed/panels/daily_panel_k5.csv
    ../../../Data/CDS_Climate/Processed/panels/daily_panel_k10.csv

James Note: in the KNN, please use k=[3,5,10]; generate three different output files with each k (file name indicating k)

Note for using for other villages:
there is a good amount of flexibility for header names
village id can be id, village_id, or pseudo_village_id and also it automatically converts to lowercase
latitude can be latitude or lat and automatically converts to lowercase
longitude can be longitude or lon and automatically converts to lowercase

it renames it later to consistent names and also if it cannot find it then there will be an error message
'''

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree  # haversine distance

# ---- Paths ----
DAILY_FEATURES_CSV = "../../../Data/CDS_Climate/Processed/daily_features/daily_features.csv"
VILLAGES_CSV = "../../../../India_Villages/village_coordinates.csv"
OUT_DIR = "../../../Data/CDS_Climate/Processed/panels/"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load daily features ----
print("Loading daily features ...")
daily = pd.read_csv(
    DAILY_FEATURES_CSV,
    parse_dates=["date"],
    dtype_backend="pyarrow",   # <-- add this
)
#daily = pd.read_csv(DAILY_FEATURES_CSV, parse_dates=["date"])

# sanity check: required cols
need_cols = {"latitude", "longitude", "date"}
missing = need_cols - set(map(str.lower, daily.columns))
if missing:
    raise ValueError(f"daily_features missing columns like {missing}. Got: {list(daily.columns)}")

# normalize column names just in case
daily = daily.rename(columns={"Latitude": "latitude", "Longitude": "longitude", "Date": "date"})
grid_points = daily[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

# ---- Load villages ----
print("Loading villages ...")
villages = pd.read_csv(VILLAGES_CSV)

# identify id / lat / lon columns
lower_map = {c.lower(): c for c in villages.columns}
id_col = lower_map.get("id") or lower_map.get("village_id") or lower_map.get("pseudo_village_id")
lat_col = lower_map.get("latitude") or lower_map.get("lat")
lon_col = lower_map.get("longitude") or lower_map.get("lon")

if not all([id_col, lat_col, lon_col]):
    raise ValueError(f"Could not find village id/lat/lon columns. Got: {list(villages.columns)}")

villages = villages.rename(columns={
    id_col: "village_id",
    lat_col: "lat",
    lon_col: "lon",
})[["village_id", "lat", "lon"]]

villages["lat"] = pd.to_numeric(villages["lat"], errors="coerce")
villages["lon"] = pd.to_numeric(villages["lon"], errors="coerce")
villages = villages.dropna(subset=["lat", "lon"]).reset_index(drop=True)

# ---- Build BallTree ----
print("Building BallTree (haversine) on grid ...")
grid_rad = np.radians(grid_points[["latitude", "longitude"]].values)
tree = BallTree(grid_rad, metric="haversine")
vill_rad = np.radians(villages[["lat", "lon"]].values)

# ---- Loop for k = 3, 5, 10 ----
for k in [3, 5, 10]:
    print(f"\nProcessing k = {k} ...")
    dist_rad, idx = tree.query(vill_rad, k=k)

    # flatten nearest points for each village
    vill_repeat = np.repeat(villages["village_id"].values, k)
    coords = grid_points.iloc[idx.flatten()].reset_index(drop=True)
    nn_map = pd.DataFrame({
        "village_id": vill_repeat,
        "grid_lat": coords["latitude"],
        "grid_lon": coords["longitude"],
    })

    # merge daily data for all k neighbors
    merged = daily.merge(
        nn_map,
        left_on=["latitude", "longitude"],
        right_on=["grid_lat", "grid_lon"],
        how="inner",
    )

    # average across k neighbors per (village_id, date)
    value_cols = [c for c in merged.columns if c not in ["village_id", "date", "latitude", "longitude", "grid_lat", "grid_lon"]]
    daily_panel_k = (
        merged.groupby(["village_id", "date"])[value_cols]
        .mean()
        .reset_index()
        .rename(columns={"date": "timestamp"})
        .sort_values(["village_id", "timestamp"])
        .reset_index(drop=True)
    )

    # save
    out_file = os.path.join(OUT_DIR, f"daily_panel_k{k}.csv")
    daily_panel_k.to_csv(out_file, index=False)
    print(f"✓ Saved {out_file}")

print("\n✓ All KNN daily panels generated successfully.")
