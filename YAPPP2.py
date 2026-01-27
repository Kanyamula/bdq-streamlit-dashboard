import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from math import radians, cos, sin, asin, sqrt
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import hashlib

# ============================================================
# BDQ Source Authority (DEFAULT): Getty TGN
# ============================================================
TGN_SOURCE_AUTHORITY = {
    "name": "The Getty Thesaurus of Geographic Names (TGN)",
    "url": "https://www.getty.edu/research/tools/vocabularies/tgn/index.html",
    "sparql": "https://vocab.getty.edu/sparql",
    "endpoint": "https://vocab.getty.edu/sparql",
}

_TGN_ISO2_CACHE = {}
_TGN_STATE_IN_COUNTRY_CACHE = {}

# ============================================================
# Helper functions
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6371 * 1000  # meters
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


@st.cache_data
def load_country_centroids():
    data = {
        "countryCode": ["US", "CA", "MX", "BW", "ZA"],
        "decimalLatitude": [39.833333, 56.130366, 23.634501, -22.328474, -30.559482],
        "decimalLongitude": [-98.583333, -106.346771, -102.552784, 24.684866, 22.937506],
        "area_km2": [9833517, 9984670, 1972550, 581730, 1221037],
    }
    return pd.DataFrame(data)


@st.cache_data
def load_source_authority_gdf():
    # Replace with Natural Earth admin-1 polygons in production
    try:
        data = {
            "iso_a2": ["US", "CA"],
            "name": ["Kansas", "Ontario"],
            "geometry": [
                Point(-98.5, 39.5).buffer(1.0, cap_style=3),
                Point(-85.0, 51.0).buffer(2.0, cap_style=3),
            ],
        }
        return gpd.GeoDataFrame(data, crs="EPSG:4326")
    except Exception as e:
        st.error(f"Could not create mock GeoDataFrame: {e}")
        return None


def _normalize_country_name(x: str) -> str:
    return " ".join(str(x).strip().split())


def _normalize_state_name(x: str) -> str:
    return " ".join(str(x).strip().split())


def _normalize_iso2(x: str) -> str:
    return str(x).strip().upper()


@st.cache_data
def tgn_available(timeout=8) -> bool:
    """External prerequisite check: can we reach the Getty SPARQL endpoint?"""
    try:
        r = requests.get(TGN_SOURCE_AUTHORITY["endpoint"], params={"query": "ASK{}"}, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _get_iso2_codes_from_tgn(country_name: str, timeout=20):
    country_name = _normalize_country_name(country_name)
    if not country_name:
        return set()

    cache_key = country_name.lower()
    if cache_key in _TGN_ISO2_CACHE:
        return _TGN_ISO2_CACHE[cache_key]

    query = f"""
    PREFIX gvp:   <http://vocab.getty.edu/ontology#>
    PREFIX skosxl:<http://www.w3.org/2008/05/skos-xl#>

    SELECT DISTINCT ?iso2
    WHERE {{
      ?place gvp:prefLabelGVP ?plabel .
      ?plabel skosxl:literalForm ?name .
      FILTER(LCASE(STR(?name)) = LCASE("{country_name}")) .

      ?place skosxl:altLabel ?codeLabel .
      ?codeLabel gvp:termKind <http://vocab.getty.edu/term/kind/ISOalpha2> .
      ?codeLabel skosxl:literalForm ?iso2 .
    }}
    """

    codes = set()
    try:
        r = requests.get(
            TGN_SOURCE_AUTHORITY["endpoint"],
            params={"query": query, "format": "application/sparql-results+json"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        codes = {
            _normalize_iso2(b["iso2"]["value"])
            for b in data.get("results", {}).get("bindings", [])
            if "iso2" in b and b["iso2"].get("value")
        }
    except (requests.RequestException, ValueError):
        codes = set()

    _TGN_ISO2_CACHE[cache_key] = codes
    return codes


def _tgn_country_state_unambiguous(country_name: str, state_name: str, timeout=25) -> bool:
    country_name = _normalize_country_name(country_name)
    state_name = _normalize_state_name(state_name)
    if not country_name or not state_name:
        return False

    cache_key = (country_name.lower(), state_name.lower())
    if cache_key in _TGN_STATE_IN_COUNTRY_CACHE:
        return _TGN_STATE_IN_COUNTRY_CACHE[cache_key]

    query = f"""
    PREFIX gvp:   <http://vocab.getty.edu/ontology#>
    PREFIX skosxl:<http://www.w3.org/2008/05/skos-xl#>

    ASK WHERE {{
      ?country gvp:prefLabelGVP ?cLabel .
      ?cLabel skosxl:literalForm ?cName .
      FILTER(LCASE(STR(?cName)) = LCASE("{country_name}")) .

      ?state gvp:prefLabelGVP ?sLabel .
      ?sLabel skosxl:literalForm ?sName .
      FILTER(LCASE(STR(?sName)) = LCASE("{state_name}")) .

      ?state gvp:broaderPreferred+ ?country .
    }}
    """

    ok = False
    try:
        r = requests.get(
            TGN_SOURCE_AUTHORITY["endpoint"],
            params={"query": query, "format": "application/sparql-results+json"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        ok = bool(data.get("boolean", False))
    except (requests.RequestException, ValueError):
        ok = False

    _TGN_STATE_IN_COUNTRY_CACHE[cache_key] = ok
    return ok


# ============================================================
# BDQ Tests (subset)
# ============================================================
def check_coordinates_center_of_country(row, buffer_meters=5000, country_centroids=None):
    if country_centroids is None or country_centroids.empty:
        return "EXTERNAL_PREREQUISITES_NOT_MET"

    country_code = str(row.get("countryCode", "")).strip().upper()
    try:
        lat = float(row.get("decimalLatitude", ""))
        lon = float(row.get("decimalLongitude", ""))
    except (ValueError, TypeError):
        return "INTERNAL_PREREQUISITES_NOT_MET"

    if not country_code or pd.isna(lat) or pd.isna(lon):
        return "INTERNAL_PREREQUISITES_NOT_MET"

    country_data = country_centroids[country_centroids["countryCode"] == country_code]
    if len(country_data) == 0:
        return "EXTERNAL_PREREQUISITES_NOT_MET"

    centroid_lat = country_data.iloc[0]["decimalLatitude"]
    centroid_lon = country_data.iloc[0]["decimalLongitude"]
    country_area = country_data.iloc[0]["area_km2"]

    distance = haversine_distance(lat, lon, centroid_lat, centroid_lon)

    try:
        uncertainty = float(row.get("coordinateUncertaintyInMeters", ""))
    except (ValueError, TypeError):
        uncertainty = None

    area_threshold = 0.5 * np.sqrt(country_area * 1_000_000)

    if distance <= buffer_meters:
        if uncertainty is None or uncertainty < area_threshold:
            return "POTENTIAL_ISSUE"

    return "NOT_ISSUE"


def test_country_not_empty(country_series, country_code_series):
    results = []
    for country, country_code in zip(country_series, country_code_series):
        country_empty = pd.isna(country) or str(country).strip() == ""
        country_code_val = str(country_code).strip().upper() if pd.notna(country_code) else ""
        country_val = str(country).strip().lower() if pd.notna(country) else ""

        if not country_empty:
            results.append("COMPLIANT")
        elif country_code_val == "XZ" and (country_empty or country_val == "high seas"):
            results.append("COMPLIANT")
        else:
            results.append("NOT_COMPLIANT")
    return pd.Series(results, index=country_series.index)


def test_coordinates_countrycode_consistent(lat_series, lon_series, country_code_series, gdf=None):
    if gdf is None:
        return pd.Series(["EXTERNAL_PREREQUISITES_NOT_MET"] * len(lat_series), index=lat_series.index)

    results = []
    for lat, lon, cc in zip(lat_series, lon_series, country_code_series):
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(cc) or str(cc).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        point = Point(lon, lat)
        match = gdf[gdf.contains(point)]
        if match.empty:
            results.append("NOT_COMPLIANT")
        elif str(cc).strip().upper() in match["iso_a2"].values:
            results.append("COMPLIANT")
        else:
            results.append("NOT_COMPLIANT")

    return pd.Series(results, index=lat_series.index)


def test_coordinates_stateprovince_consistent(lat_series, lon_series, state_series, gdf=None):
    if gdf is None:
        return pd.Series(["EXTERNAL_PREREQUISITES_NOT_MET"] * len(lat_series), index=lat_series.index)

    results = []
    for lat, lon, state in zip(lat_series, lon_series, state_series):
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(state) or str(state).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        point = Point(lon, lat)
        match = gdf[gdf.contains(point)]
        if match.empty:
            results.append("NOT_COMPLIANT")
        elif str(state).strip().lower() in [s.lower() for s in match["name"].values]:
            results.append("COMPLIANT")
        else:
            results.append("NOT_COMPLIANT")

    return pd.Series(results, index=lat_series.index)


def test_coordinates_not_zero(lat_series, lon_series):
    results = []
    for lat, lon in zip(lat_series, lon_series):
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(lat) or pd.isna(lon):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        results.append("COMPLIANT" if ((lat != 0) or (lon != 0)) else "NOT_COMPLIANT")

    return pd.Series(results, index=lat_series.index)


def test_coordinate_uncertainty_inrange(series):
    results = []
    for val in series:
        if pd.isna(val) or str(val).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            uncertainty = float(val)
            results.append("COMPLIANT" if 1 <= uncertainty <= 20037509 else "NOT_COMPLIANT")
        except (TypeError, ValueError):
            results.append("NOT_COMPLIANT")
    return pd.Series(results, index=series.index)


def test_countrycode_not_empty(country_series, country_code_series):
    results = []
    for country, country_code in zip(country_series, country_code_series):
        country_empty = pd.isna(country) or str(country).strip() == ""
        if not country_empty:
            results.append("COMPLIANT")
        elif str(country_code).strip().upper() == "XZ" and (
            country_empty or str(country).strip().lower() == "high seas"
        ):
            results.append("COMPLIANT")
        else:
            results.append("NOT_COMPLIANT")
    return pd.Series(results, index=country_series.index)


def test_countrycode_standard(series):
    valid_country_codes = ["US", "CA", "MX", "GB", "DE", "FR", "JP", "CN", "BR", "AR", "BW", "ZA", "XZ"]
    results = []
    for val in series:
        if pd.isna(val) or str(val).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append("COMPLIANT" if str(val).strip().upper() in valid_country_codes else "NOT_COMPLIANT")
    return pd.Series(results, index=series.index)


def test_country_countrycode_consistent(country_series, countrycode_series, source_authority_available=None):
    if source_authority_available is None:
        source_authority_available = tgn_available()

    results = []
    for country, country_code in zip(country_series, countrycode_series):
        if not source_authority_available:
            results.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        country_empty = pd.isna(country) or str(country).strip() == ""
        code_empty = pd.isna(country_code) or str(country_code).strip() == ""
        if country_empty or code_empty:
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        country_norm = _normalize_country_name(country)
        code_norm = _normalize_iso2(country_code)
        iso2_set = _get_iso2_codes_from_tgn(country_norm)

        results.append("COMPLIANT" if code_norm in iso2_set else "NOT_COMPLIANT")

    return pd.Series(results, index=country_series.index)


def test_country_stateprovince_unambiguous(country_series, state_series, source_authority_available=None):
    if source_authority_available is None:
        source_authority_available = tgn_available()

    results = []
    for country, state in zip(country_series, state_series):
        if not source_authority_available:
            results.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        country_empty = pd.isna(country) or str(country).strip() == ""
        state_empty = pd.isna(state) or str(state).strip() == ""
        if country_empty or state_empty:
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        ok = _tgn_country_state_unambiguous(str(country), str(state))
        results.append("COMPLIANT" if ok else "NOT_COMPLIANT")

    return pd.Series(results, index=country_series.index)


def test_country_found(series):
    return pd.Series(
        ["COMPLIANT" if (pd.notna(v) and str(v).strip() != "") else "NOT_COMPLIANT" for v in series],
        index=series.index,
    )


def test_decimallatitude_inrange(lat_series):
    results = []
    for lat in lat_series:
        try:
            lat = float(lat)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append("COMPLIANT" if -90 <= lat <= 90 else "NOT_COMPLIANT")
    return pd.Series(results, index=lat_series.index)


def test_decimallatitude_notempty(lat_series):
    results = []
    for lat in lat_series:
        results.append("COMPLIANT" if (pd.notna(lat) and str(lat).strip() != "") else "NOT_COMPLIANT")
    return pd.Series(results, index=lat_series.index)


def test_decimallongitude_inrange(lon_series):
    results = []
    for lon in lon_series:
        try:
            lon = float(lon)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append("COMPLIANT" if -180 <= lon <= 180 else "NOT_COMPLIANT")
    return pd.Series(results, index=lon_series.index)


def test_decimallongitude_notempty(lon_series):
    results = []
    for lon in lon_series:
        results.append("COMPLIANT" if (pd.notna(lon) and str(lon).strip() != "") else "NOT_COMPLIANT")
    return pd.Series(results, index=lon_series.index)


def test_location_notempty(df_row):
    location_columns = [
        "higherGeographyID", "higherGeography", "continent", "country",
        "countryCode", "stateProvince", "county", "municipality",
        "waterBody", "island", "islandGroup", "locality", "locationID",
        "verbatimLocality", "decimalLatitude", "decimalLongitude",
        "verbatimCoordinates", "verbatimLatitude", "verbatimLongitude",
        "footprintWKT",
    ]
    if any(pd.notna(df_row.get(col, pd.NA)) and str(df_row.get(col, "")).strip() != "" for col in location_columns):
        return "COMPLIANT"
    return "NOT_COMPLIANT"


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(layout="wide")
st.title("AFRICAN TROPICAL PLANTS Geospatial Data Quality Validator")
st.caption("Upload a CSV file with geospatial data for BDQ-style validation.")

gdf_states = load_source_authority_gdf()
country_centroids = load_country_centroids()
tgn_is_up = tgn_available()

# ---- Upload history: date of uploading a version + its status (ONLY) ----
if "upload_history" not in st.session_state:
    st.session_state["upload_history"] = []  # list of dicts: uploaded_at, version_id, status

def _file_version_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:12]

def _dataset_status_from_summary(summary_df: pd.DataFrame) -> str:
    """
    Single status for the uploaded dataset version.
    Priority:
      1) Any EXTERNAL/INTERNAL prereq not met -> PREREQUISITES_NOT_MET
      2) Any NOT_COMPLIANT or POTENTIAL_ISSUE -> ISSUES_FOUND
      3) Otherwise -> ALL_COMPLIANT
    """
    if summary_df.empty:
        return "UNKNOWN"

    prereq = summary_df["Prerequisite Not Met"].sum() if "Prerequisite Not Met" in summary_df.columns else 0
    not_ok = summary_df["Not Compliant Count"].sum() if "Not Compliant Count" in summary_df.columns else 0
    pot = summary_df["Potential Issue Count"].sum() if "Potential Issue Count" in summary_df.columns else 0

    if prereq > 0:
        return "PREREQUISITES_NOT_MET"
    if (not_ok + pot) > 0:
        return "ISSUES_FOUND"
    return "ALL_COMPLIANT"


with st.sidebar:
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    st.markdown("---")
    st.header("Source Authority Status")
    st.markdown(f"**Natural Earth Admin-1 (Mock):** {'✅ Loaded' if gdf_states is not None else '❌ Not Loaded'}")
    st.markdown(f"**Country Centroids (Mock):** {'✅ Loaded' if country_centroids is not None else '❌ Not Loaded'}")
    st.markdown(
        f"**Getty TGN (SPARQL):** {'✅ Available' if tgn_is_up else '❌ Not available'}  \n"
        f"{TGN_SOURCE_AUTHORITY['url']}"
    )
    st.markdown("---")
    st.caption("If TGN is not available, tests that depend on TGN will return EXTERNAL_PREREQUISITES_NOT_MET.")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin the data quality validation.")
    st.stop()

# Load data
try:
    raw_bytes = uploaded_file.getvalue()
    version_id = _file_version_id(raw_bytes)
    stringio = StringIO(raw_bytes.decode("cp1252"))
    df = pd.read_csv(stringio)
except Exception as e:
    st.error(f"Error loading or decoding file: {e}")
    st.stop()

st.subheader("Original Data Preview")
st.dataframe(df.head())

REQUIRED_COLS = [
    "country", "countryCode", "decimalLatitude", "decimalLongitude",
    "stateProvince", "coordinateUncertaintyInMeters", "geodeticDatum",
    "minimumElevationInMeters", "maximumElevationInMeters",
]
for col in REQUIRED_COLS:
    if col not in df.columns:
        df[col] = np.nan


@st.cache_data
def run_all_tests(data_df, _states_gdf, centroids_df, _tgn_is_available: bool):
    df_result = data_df.copy()

    df_result["DQ_COUNTRY_NOT_EMPTY"] = test_country_not_empty(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("countryCode", pd.Series(dtype="object")),
    )

    if _states_gdf is not None:
        df_result["DQ_COORDINATES_COUNTRYCODE_CONSISTENT"] = test_coordinates_countrycode_consistent(
            df_result["decimalLatitude"],
            df_result["decimalLongitude"],
            df_result.get("countryCode", pd.Series(dtype="object")),
            gdf=_states_gdf,
        )
        df_result["DQ_COORDINATES_STATEPROVINCE_CONSISTENT"] = test_coordinates_stateprovince_consistent(
            df_result["decimalLatitude"],
            df_result["decimalLongitude"],
            df_result.get("stateProvince", pd.Series(dtype="object")),
            gdf=_states_gdf,
        )
    else:
        df_result["DQ_COORDINATES_COUNTRYCODE_CONSISTENT"] = "EXTERNAL_PREREQUISITES_NOT_MET"
        df_result["DQ_COORDINATES_STATEPROVINCE_CONSISTENT"] = "EXTERNAL_PREREQUISITES_NOT_MET"

    df_result["DQ_COORDINATES_NOTZERO"] = test_coordinates_not_zero(
        df_result["decimalLatitude"],
        df_result["decimalLongitude"],
    )

    df_result["DQ_COORDINATEUNCERTAINTY_INRANGE"] = test_coordinate_uncertainty_inrange(
        df_result.get("coordinateUncertaintyInMeters", pd.Series(dtype="object"))
    )

    df_result["DQ_COUNTRYCODE_NOTEMPTY"] = test_countrycode_not_empty(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("countryCode", pd.Series(dtype="object")),
    )

    df_result["DQ_COUNTRYCODE_STANDARD"] = test_countrycode_standard(
        df_result.get("countryCode", pd.Series(dtype="object"))
    )

    # TGN-dependent
    df_result["DQ_COUNTRY_COUNTRYCODE_CONSISTENT"] = test_country_countrycode_consistent(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("countryCode", pd.Series(dtype="object")),
        source_authority_available=_tgn_is_available,
    )

    df_result["DQ_COUNTRY_STATEPROVINCE_UNAMBIGUOUS"] = test_country_stateprovince_unambiguous(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("stateProvince", pd.Series(dtype="object")),
        source_authority_available=_tgn_is_available,
    )

    df_result["DQ_COUNTRY_FOUND"] = test_country_found(df_result.get("country", pd.Series(dtype="object")))
    df_result["DQ_DECIMALLATITUDE_INRANGE"] = test_decimallatitude_inrange(df_result["decimalLatitude"])
    df_result["DQ_DECIMALLATITUDE_NOTEMPTY"] = test_decimallatitude_notempty(df_result["decimalLatitude"])
    df_result["DQ_DECIMALLONGITUDE_INRANGE"] = test_decimallongitude_inrange(df_result["decimalLongitude"])
    df_result["DQ_DECIMALLONGITUDE_NOTEMPTY"] = test_decimallongitude_notempty(df_result["decimalLongitude"])
    df_result["DQ_LOCATION_NOTEMPTY"] = df_result.apply(test_location_notempty, axis=1)

    df_result["DQ_ISSUE_COORDINATES_CENTEROFCOUNTRY"] = df_result.apply(
        lambda row: check_coordinates_center_of_country(row, buffer_meters=5000, country_centroids=centroids_df),
        axis=1,
    )

    return df_result


with st.spinner("Running Data Quality Validation..."):
    validated_df = run_all_tests(df, gdf_states, country_centroids, tgn_is_up)

# ============================================================
# Results + Summary
# ============================================================
st.markdown("---")
st.subheader("Data Quality Validation Results")

dq_cols = [c for c in validated_df.columns if c.startswith("DQ_")]
summary = pd.DataFrame(
    {
        "Test": dq_cols,
        "Compliant Count": [validated_df[c].eq("COMPLIANT").sum() for c in dq_cols],
        "Not Compliant Count": [validated_df[c].eq("NOT_COMPLIANT").sum() for c in dq_cols],
        "Potential Issue Count": [validated_df[c].eq("POTENTIAL_ISSUE").sum() for c in dq_cols],
        "Prerequisite Not Met": [
            validated_df[c].isin(["INTERNAL_PREREQUISITES_NOT_MET", "EXTERNAL_PREREQUISITES_NOT_MET"]).sum()
            for c in dq_cols
        ],
    }
).set_index("Test")

st.dataframe(summary)

# ============================================================
# Register this upload "version" in history (date + status only)
# ============================================================
uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
dataset_status = _dataset_status_from_summary(summary)

# Avoid duplicating identical file uploads
already = any(h.get("version_id") == version_id for h in st.session_state["upload_history"])
if not already:
    st.session_state["upload_history"].append(
        {"uploaded_at": uploaded_at, "version_id": version_id, "status": dataset_status}
    )

# Full results
st.markdown("### Full DataFrame with Validation Columns")
st.dataframe(validated_df)

@st.cache_data
def convert_df(df_):
    return df_.to_csv(index=False).encode("cp1252")

st.download_button(
    label="Download Validated Data (.csv)",
    data=convert_df(validated_df),
    file_name="validated_data_quality_results.csv",
    mime="text/csv",
)

# ============================================================
# Stacked bar chart (COMPLETE)
# ============================================================
st.markdown("---")
st.markdown("### Stacked bar chart of BDQ test outcomes")

stack_cols = ["Compliant Count", "Not Compliant Count", "Potential Issue Count", "Prerequisite Not Met"]
all_tests = summary.index.tolist()
default_tests = all_tests[: min(25, len(all_tests))]

selected_tests_for_bar = st.multiselect(
    "Select tests to include in the stacked bar chart:",
    options=all_tests,
    default=default_tests,
    key="bar_tests",
)

plot_summary = summary.loc[selected_tests_for_bar, stack_cols].copy()
plot_long = plot_summary.reset_index().melt(id_vars="Test", var_name="Outcome", value_name="Count")

sort_mode = st.selectbox(
    "Sort tests by:",
    options=["Original order", "Total issues (Not Compliant + Potential Issue)", "Total records (All outcomes)"],
    index=1,
    key="bar_sort",
)

if sort_mode == "Total issues (Not Compliant + Potential Issue)":
    totals = summary["Not Compliant Count"] + summary["Potential Issue Count"]
    order = totals.loc[selected_tests_for_bar].sort_values(ascending=False).index.tolist()
    plot_long["Test"] = pd.Categorical(plot_long["Test"], categories=order, ordered=True)

elif sort_mode == "Total records (All outcomes)":
    totals = summary[stack_cols].sum(axis=1)
    order = totals.loc[selected_tests_for_bar].sort_values(ascending=False).index.tolist()
    plot_long["Test"] = pd.Categorical(plot_long["Test"], categories=order, ordered=True)

fig_bar = px.bar(
    plot_long,
    x="Test",
    y="Count",
    color="Outcome",
    barmode="stack",
    title="BDQ Outcomes by Test (Stacked Bar)",
)

fig_bar.update_layout(
    xaxis_title="Test",
    yaxis_title="Count",
    xaxis_tickangle=-45,
    height=650,
    margin={"r": 0, "t": 60, "l": 0, "b": 0},
)

st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# Interactive "dataCleaning-style" dashboard
# Requirement:
#   - show ONLY upload date (dataset version) + status of each test
#   - also provide a map of points colored by selected test status
#
# Assumptions / required session_state structure:
#   st.session_state["upload_history"] is a list of dicts like:
#     {
#       "version_id": "v20260126_221530",
#       "uploaded_at": "2026-01-26 22:15:30",   # or datetime
#       "test_status": {                         # status per test for this upload version
#           "DQ_COUNTRY_NOT_EMPTY": "COMPLIANT",
#           "DQ_COUNTRY_COUNTRYCODE_CONSISTENT": "NOT_COMPLIANT",
#           ...
#       }
#     }
#
# If you do not already create this, you MUST add the "Append to upload history"
# block shown below right after you compute validated_df and summary.
# ============================================================

# ============================
# (A) Append to upload history (counts per test per upload)
# ============================

SUSPECT_STATUSES_DEFAULT = [
    "NOT_COMPLIANT",
    "POTENTIAL_ISSUE",
    "INTERNAL_PREREQUISITES_NOT_MET",
    "EXTERNAL_PREREQUISITES_NOT_MET",
]

# Ensure container exists
if "upload_history" not in st.session_state:
    st.session_state["upload_history"] = []

# Ensure current version metadata exists
if "current_version_id" not in st.session_state:
    st.session_state["current_version_id"] = pd.Timestamp.now().strftime("v%Y%m%d_%H%M%S")

if "current_uploaded_at" not in st.session_state:
    st.session_state["current_uploaded_at"] = pd.Timestamp.now()

# Identify DQ test columns
dq_cols_now = [
    c for c in validated_df.columns
    if c.startswith("DQ_") and not c.startswith("DQ_COMMENT_")
]

# Compute per-test counts for THIS upload
test_counts = {}
for test_col in dq_cols_now:
    s = validated_df[test_col].astype(str)
    test_counts[test_col] = {
        "suspect_records": int(s.isin(SUSPECT_STATUSES_DEFAULT).sum()),
        "total_records": int(len(s)),
        "COMPLIANT": int((s == "COMPLIANT").sum()),
        "NOT_COMPLIANT": int((s == "NOT_COMPLIANT").sum()),
        "POTENTIAL_ISSUE": int((s == "POTENTIAL_ISSUE").sum()),
        "INTERNAL_PREREQUISITES_NOT_MET": int((s == "INTERNAL_PREREQUISITES_NOT_MET").sum()),
        "EXTERNAL_PREREQUISITES_NOT_MET": int((s == "EXTERNAL_PREREQUISITES_NOT_MET").sum()),
    }

# Avoid duplicates on rerun
already = any(h.get("version_id") == st.session_state["current_version_id"]
              for h in st.session_state["upload_history"] if isinstance(h, dict))

if not already:
    st.session_state["upload_history"].append({
        "version_id": st.session_state["current_version_id"],
        "uploaded_at": st.session_state["current_uploaded_at"],
        "test_counts": test_counts,   # <-- stored per upload
    })

# ============================================================
# Make the dashboard look like your screenshot:
#   - small-multiple “cards” (2 columns) with line+markers
#   - y-axis = Records (count of suspect records for that test in each upload)
#   - x-axis = upload date (each dataset version)
#
# IMPORTANT:
# Your current upload_history only stores per-test "dominant status".
# To draw curves like your screenshot, you must also store per-test COUNTS
# per upload (e.g., number of suspect records for that test).
#
# 1) ADD/REPLACE the "append to upload_history" block right after validated_df is created.
# 2) REPLACE your Section (B) with the dashboard code below.
# ============================================================


# ============================================================
# (1) ADD THIS RIGHT AFTER you compute `validated_df`
#     (this will store per-test counts per upload version)
# ============================================================

# Define which outcomes are considered "suspect" (like dataCleaning)
SUSPECT_STATUSES_DEFAULT = [
    "NOT_COMPLIANT",
    "POTENTIAL_ISSUE",
    "INTERNAL_PREREQUISITES_NOT_MET",
    "EXTERNAL_PREREQUISITES_NOT_MET",
]

# Make sure these exist
if "upload_history" not in st.session_state:
    st.session_state["upload_history"] = []

if "current_version_id" not in st.session_state:
    st.session_state["current_version_id"] = pd.Timestamp.now().strftime("v%Y%m%d_%H%M%S")

if "current_uploaded_at" not in st.session_state:
    st.session_state["current_uploaded_at"] = pd.Timestamp.now()

# Compute per-test suspect counts for THIS upload
dq_cols_now = [
    c for c in validated_df.columns
    if c.startswith("DQ_") and not c.startswith("DQ_COMMENT_")
]

test_counts = {}
for test_col in dq_cols_now:
    s = validated_df[test_col].astype(str)
    test_counts[test_col] = {
        "suspect_records": int(s.isin(SUSPECT_STATUSES_DEFAULT).sum()),
        "total_records": int(len(s)),
        # (optional) store breakdown if you want later
        "COMPLIANT": int((s == "COMPLIANT").sum()),
        "NOT_COMPLIANT": int((s == "NOT_COMPLIANT").sum()),
        "POTENTIAL_ISSUE": int((s == "POTENTIAL_ISSUE").sum()),
        "INTERNAL_PREREQUISITES_NOT_MET": int((s == "INTERNAL_PREREQUISITES_NOT_MET").sum()),
        "EXTERNAL_PREREQUISITES_NOT_MET": int((s == "EXTERNAL_PREREQUISITES_NOT_MET").sum()),
    }

# Avoid duplicates on rerun
already = any(h.get("version_id") == st.session_state["current_version_id"] for h in st.session_state["upload_history"])
if not already:
    st.session_state["upload_history"].append({
        "version_id": st.session_state["current_version_id"],
        "uploaded_at": st.session_state["current_uploaded_at"],
        # optional label like in the screenshot
        "collection": st.session_state.get("collection", "AFR"),
        "test_counts": test_counts,  # <-- THIS enables the curves
    })


# ============================================================
# (2) REPLACE YOUR SECTION (B) WITH THIS DASHBOARD
# ============================================================

st.markdown("---")
# If you have a “collection” label, show it like the screenshot
collection_label = st.session_state.get("collection", "")
st.markdown(f"**collection: {collection_label}**")

st.header("dataCleaning — Suspect records over upload dates")
# ============================================================
# One-time migration: add test_counts to older history entries
# ============================================================
if "upload_history" in st.session_state and isinstance(st.session_state["upload_history"], list):
    migrated = 0

    for h in st.session_state["upload_history"]:
        # If entry already has test_counts, skip
        if isinstance(h, dict) and "test_counts" in h:
            continue

        # If entry has per-test status dict, convert it to count-like data
        # (Suspect = NOT_COMPLIANT, POTENTIAL_ISSUE, INTERNAL..., EXTERNAL...)
        ts = h.get("test_status", None) if isinstance(h, dict) else None
        if isinstance(ts, dict) and len(ts) > 0:
            suspect_set = {
                "NOT_COMPLIANT",
                "POTENTIAL_ISSUE",
                "INTERNAL_PREREQUISITES_NOT_MET",
                "EXTERNAL_PREREQUISITES_NOT_MET",
            }

            # We do NOT have record-level counts for old versions.
            # So we approximate: suspect_records = 1 if that upload's status is suspect, else 0.
            test_counts = {}
            for test, status in ts.items():
                status = str(status)
                test_counts[str(test)] = {
                    "suspect_records": 1 if status in suspect_set else 0,
                    "total_records": 1,
                    "COMPLIANT": 1 if status == "COMPLIANT" else 0,
                    "NOT_COMPLIANT": 1 if status == "NOT_COMPLIANT" else 0,
                    "POTENTIAL_ISSUE": 1 if status == "POTENTIAL_ISSUE" else 0,
                    "INTERNAL_PREREQUISITES_NOT_MET": 1 if status == "INTERNAL_PREREQUISITES_NOT_MET" else 0,
                    "EXTERNAL_PREREQUISITES_NOT_MET": 1 if status == "EXTERNAL_PREREQUISITES_NOT_MET" else 0,
                }

            h["test_counts"] = test_counts
            migrated += 1

    if migrated > 0:
        st.info(f"Migrated {migrated} historical uploads to include approximate test_counts.")



history = st.session_state.get("upload_history", [])
history_df = pd.DataFrame(history)

if history_df.empty:
    st.info("No uploads recorded yet.")
    st.stop()

# Parse timestamps
history_df["uploaded_at"] = pd.to_datetime(history_df["uploaded_at"], errors="coerce")
history_df = history_df.dropna(subset=["uploaded_at"]).sort_values("uploaded_at")

# Expand test_counts to long format: (uploaded_at, version_id, test, suspect_records)
long_rows = []
for _, r in history_df.iterrows():
    tc = r.get("test_counts", None)

    # HARDENING
    if tc is None or isinstance(tc, float) or not isinstance(tc, dict):
        continue

    for test, metrics in tc.items():
        if not isinstance(metrics, dict):
            continue
        long_rows.append({
            "uploaded_at": r["uploaded_at"],
            "version_id": r.get("version_id", ""),
            "test": str(test),
            "suspect_records": int(metrics.get("suspect_records", 0)),
            "total_records": int(metrics.get("total_records", 0)),
        })

long_df = pd.DataFrame(long_rows)

if long_df.empty:
    st.warning(
        "Upload history exists, but contains no per-test count data. "
        "Confirm you added the `test_counts` storage block right after `validated_df`."
    )
    st.stop()

# ---------------------------
# Optional: compact table (like a version log)
# ---------------------------
with st.expander("Show upload versions (date + dataset version only)"):
    tbl = history_df[["uploaded_at", "version_id"]].copy()
    tbl.rename(columns={"uploaded_at": "Upload date", "version_id": "Dataset version"}, inplace=True)
    st.dataframe(tbl, use_container_width=True)

# ---------------------------
# Controls (which tests to show)
# ---------------------------
st.subheader("Panels")
all_tests = sorted(long_df["test"].dropna().unique().tolist())

# Default: first 6 tests
default_tests = all_tests[: min(6, len(all_tests))]

c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
with c1:
    panel_tests = st.multiselect(
        "Select tests to display as panels:",
        options=all_tests,
        default=default_tests,
        key="dc_panel_tests",
    )
with c2:
    show_cumulative = st.checkbox("Cumulative", value=False, key="dc_cumulative")
with c3:
    use_log = st.checkbox("Log y-axis", value=False, key="dc_logy")

if len(panel_tests) == 0:
    st.info("Select at least one test to display.")
    st.stop()

# ---------------------------
# Styling to resemble the screenshot “cards”
# ---------------------------
st.markdown(
    """
    <style>
      .dc-card {
        border: 1px solid #d9d9d9;
        background: #f6f6f6;
        padding: 8px 10px 0px 10px;
        border-radius: 2px;
        margin-bottom: 14px;
      }
      .dc-title {
        font-weight: 600;
        text-align: center;
        margin: 2px 0 6px 0;
        color: #3a3a3a;
        text-transform: lowercase;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper: build one panel chart (orange line + open circle markers)
# ---------------------------
def make_panel_chart(df_in: pd.DataFrame, test_name: str, cumulative: bool, logy: bool):
    d = df_in[df_in["test"] == test_name].sort_values("uploaded_at").copy()

    if d.empty:
        # Return an empty figure with a helpful message rather than erroring
        fig = go.Figure()
        fig.update_layout(
            height=270,
            margin=dict(l=40, r=10, t=10, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f2efee",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No data for this test.", x=0.5, y=0.5, showarrow=False)],
        )
        return fig

    if cumulative:
        d["y"] = d["suspect_records"].cumsum()
    else:
        d["y"] = d["suspect_records"]

    line_color = "#d9792d"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["uploaded_at"],
            y=d["y"],
            mode="lines+markers",
            line=dict(width=3, color=line_color),
            marker=dict(size=7, symbol="circle-open", line=dict(width=2, color=line_color)),
            hovertemplate=(
                "Upload: %{x|%Y-%m-%d %H:%M}<br>"
                "Suspect records: %{y}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=270,
        margin=dict(l=40, r=10, t=10, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f2efee",
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#eadbd4",
            tickangle=-55,
            tickformat="%d-%m-%Y",
        ),
        yaxis=dict(
            title="Records",
            showgrid=True,
            gridcolor="#eadbd4",
            type="log" if logy else "linear",
        ),
        showlegend=False,
    )
    return fig

# ---------------------------
# Render as 2-column grid like screenshot
# ---------------------------
cols = st.columns(2, vertical_alignment="top")

for i, t in enumerate(panel_tests):
    title = t.replace("DQ_", "").replace("_", " ").lower()

    with cols[i % 2]:
        st.markdown(
            f'<div class="dc-card"><div class="dc-title">{title}</div>',
            unsafe_allow_html=True
        )

        fig = make_panel_chart(long_df, t, cumulative=show_cumulative, logy=use_log)

        # ✅ UNIQUE KEY per panel, stable across reruns and UI toggles
        panel_key = f"dc_panel_plot_{t}_{i}_cum{int(show_cumulative)}_log{int(use_log)}"
        st.plotly_chart(fig, use_container_width=True, key=panel_key)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# (3) Map of points with different colours based on test status
# ============================================================
st.markdown("---")
st.header("Map — Points colored by selected test status")

dq_cols_for_map = [
    c for c in validated_df.columns
    if c.startswith("DQ_") and not c.startswith("DQ_COMMENT_")
]

m1, m2 = st.columns([1, 2], vertical_alignment="top")

with m1:
    selected_test_map = st.selectbox(
        "Select a test for map coloring:",
        options=dq_cols_for_map,
        index=0,
        key="dc_map_selected_test"
    )

    status_order = [
        "COMPLIANT",
        "POTENTIAL_ISSUE",
        "NOT_COMPLIANT",
        "INTERNAL_PREREQUISITES_NOT_MET",
        "EXTERNAL_PREREQUISITES_NOT_MET",
        "NOT_ISSUE",
    ]

    map_statuses = [
        s for s in status_order
        if s in validated_df[selected_test_map].astype(str).unique().tolist()
    ]

    selected_map_statuses = st.multiselect(
        "Filter statuses to show on map:",
        options=map_statuses,
        default=map_statuses,
        key="dc_map_statuses"
    )

with m2:
    if "decimalLatitude" not in validated_df.columns or "decimalLongitude" not in validated_df.columns:
        st.error("Map requires columns: decimalLatitude and decimalLongitude.")
    else:
        map_df = validated_df.copy()
        map_df["decimalLatitude"] = pd.to_numeric(map_df["decimalLatitude"], errors="coerce")
        map_df["decimalLongitude"] = pd.to_numeric(map_df["decimalLongitude"], errors="coerce")
        map_df = map_df.dropna(subset=["decimalLatitude", "decimalLongitude"])

        if selected_map_statuses:
            map_df = map_df[map_df[selected_test_map].astype(str).isin(selected_map_statuses)]

        if map_df.empty:
            st.info("No points to display after filtering.")
        else:
            color_map = {
                "COMPLIANT": "#2ecc71",
                "NOT_COMPLIANT": "#e74c3c",
                "POTENTIAL_ISSUE": "#f1c40f",
                "INTERNAL_PREREQUISITES_NOT_MET": "#95a5a6",
                "EXTERNAL_PREREQUISITES_NOT_MET": "#34495e",
                "NOT_ISSUE": "#3498db",
            }

            test_short = selected_test_map.replace("DQ_", "")
            comment_col = f"DQ_COMMENT_{test_short}"

            hover_data = {
                "decimalLatitude": True,
                "decimalLongitude": True,
                selected_test_map: True,
            }
            if comment_col in map_df.columns:
                hover_data[comment_col] = True

            fig_map = px.scatter_mapbox(
                map_df,
                lat="decimalLatitude",
                lon="decimalLongitude",
                color=selected_test_map,
                color_discrete_map=color_map,
                hover_name="country" if "country" in map_df.columns else None,
                hover_data=hover_data,
                zoom=2,
                height=650,
                title=f"Points colored by: {selected_test_map}",
            )
            fig_map.update_layout(
                mapbox_style="carto-positron",
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
            )

            # ✅ UNIQUE KEY per map state (test + statuses)
            statuses_key_part = "_".join(selected_map_statuses) if selected_map_statuses else "ALL"
            map_key = f"dc_map_{selected_test_map}_{statuses_key_part}"
            st.plotly_chart(fig_map, use_container_width=True, key=map_key)
