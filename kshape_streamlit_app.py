import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape
from tslearn.utils import to_time_series_dataset

st.set_page_config(page_title="Zeitreihen-Clustering", layout="centered")

st.title("Clustering für RELEX")
st.markdown("Testversion.")

# -----------------------------
# Parameter
# -----------------------------
MIN_DATA_RATIO = 0.20     # 20%-Regel
ZERO_EPS = 1e-12          # Toleranz für "ist 0?"
RANDOM_STATE = 42

# -----------------------------
# Helper
# -----------------------------
def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def _preprocess_dynamic(df_wide: pd.DataFrame, min_data_ratio: float = 0.2):
    series_list, index_map, special_ids = [], [], []
    weeks = list(df_wide.columns)
    for ridx, row in df_wide.iterrows():
        raw = row.values.astype(float)
        vals_for_null = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        non_zero_mask = np.abs(vals_for_null) > ZERO_EPS
        data_ratio = non_zero_mask.sum() / len(vals_for_null)
        if data_ratio <= min_data_ratio:
            special_ids.append(ridx)
            continue
        nz_idx = np.where(non_zero_mask)[0]
        if nz_idx.size == 0:
            special_ids.append(ridx)
            continue
        start, end = nz_idx[0], nz_idx[-1] + 1
        trimmed = raw[start:end]
        s = pd.Series(trimmed, dtype="float64").interpolate(method="linear", limit_direction="both")
        s = s.fillna(0.0)
        if (np.abs(s.values) <= ZERO_EPS).all():
            special_ids.append(ridx)
            continue
        series_list.append(s.values.astype(float))
        index_map.append(ridx)
    return series_list, index_map, special_ids, weeks

def _plot_cluster_block(weeks, X_scaled, labels, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    unique_labels = sorted(np.unique(labels))
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        for i in idxs:
            ax.plot(range(X_scaled.shape[1]), X_scaled[i].ravel(), alpha=0.2)
        mean_curve = np.nanmean(X_scaled[idxs], axis=0).ravel()
        ax.plot(range(X_scaled.shape[1]), mean_curve, color="black", linewidth=2, label=f"Cluster {lab}")
    ax.set_title(title)
    ax.set_xlabel("Kalenderwoche (Index)")
    ax.set_ylabel("Z-transformierter Absatz")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# UI: Clusterwahl
# -----------------------------
cluster_option = st.radio("Clusteranzahl bestimmen:", ["Automatisch (Elbow)", "Manuell wählen"])
if cluster_option == "Manuell wählen":
    user_defined_k = st.slider("Wähle Clusteranzahl (k)", 2, 10, 3)
else:
    user_defined_k = None

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx"])
if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    df_raw.rename(columns={df_raw.columns[0]: "Markt"}, inplace=True)
    df_raw["Markt"] = df_raw["Markt"].astype(str).str.strip()
    df_raw.set_index("Markt", inplace=True)
    df_num = _safe_numeric(df_raw)
    st.success(f"{len(df_num)} Märkte geladen.")

    # ALT-Variante (Nullphasen drin)
    X_old = df_num.fillna(0.0).values
    scaler_old = TimeSeriesScalerMeanVariance()
    X_old_scaled = scaler_old.fit_transform(X_old)
    kshape_old = None
    labels_old = None

    # NEU-Variante (Nullphasen raus, 20%-Regel)
    series_list, idx_map, special_ids, weeks = _preprocess_dynamic(df_num, MIN_DATA_RATIO)
    if len(series_list) > 0:
        X_ts = to_time_series_dataset(series_list)
        scaler_new = TimeSeriesScalerMeanVariance()
        X_new_scaled = scaler_new.fit_transform(X_ts)
        kshape_new = None
        labels_new_core = None
        full_labels_new = None
    else:
        X_new_scaled = None
        full_labels_new = pd.Series("Sondercluster (zu wenig/keine Daten)", index=df_num.index)

    # Elbow für NEU-Daten
    def run_elbow(X_scaled):
        inertia, K_range = [], range(2, 10)
        for k in K_range:
            ks = KShape(n_clusters=k, random_state=RANDOM_STATE)
            ks.fit(X_scaled)
            inertia.append(ks.inertia_)
        diffs = np.diff(inertia)
        best_k_index = np.argmax(-diffs) + 1
        return K_range[best_k_index], K_range, inertia

    if user_defined_k is None:
        if X_new_scaled is not None:
            optimal_k, K_range, inertia = run_elbow(X_new_scaled)
            st.success(f"Automatisch gewählte Clusteranzahl: {optimal_k}")
            fig, ax = plt.subplots()
            ax.plot(list(K_range), inertia, marker='o')
            ax.set_title("Elbow-Methode (auf 'Neu'-Daten)")
            ax.set_xlabel("Clusteranzahl (k)")
            ax.set_ylabel("Inertia")
            st.pyplot(fig)
        else:
            st.error("Keine NEU-Daten verfügbar.")
            st.stop()
    else:
        optimal_k = user_defined_k
        st.success(f"Manuell gewählte Clusteranzahl: {optimal_k}")

    # Cluster berechnen
    kshape_old = KShape(n_clusters=optimal_k, random_state=RANDOM_STATE)
    labels_old = kshape_old.fit_predict(X_old_scaled)

    if X_new_scaled is not None:
        kshape_new = KShape(n_clusters=optimal_k, random_state=RANDOM_STATE)
        labels_new_core = kshape_new.fit_predict(X_new_scaled)
        full_labels_new = pd.Series(index=df_num.index, dtype=object)
        for mid, lab in zip(idx_map, labels_new_core):
            full_labels_new.loc[mid] = lab
        for sid in special_ids:
            full_labels_new.loc[sid] = "Sondercluster (zu wenig/keine Daten)"

    # -----------------------------
    # Schalter ALT/NEU
    # -----------------------------
    mode = st.selectbox("Ansicht wählen:", ["NEU (Nullphasen entfernt, 20%-Regel)", "ALT (Nullphasen enthalten)"])

    if mode.startswith("NEU"):
        st.subheader("Clusterplot (NEU)")
        _plot_cluster_block(weeks, X_new_scaled.squeeze(-1), labels_new_core, "NEU: Nullphasen entfernt")
        cluster_labels = full_labels_new
    else:
        st.subheader("Clusterplot (ALT)")
        _plot_cluster_block(weeks, X_old_scaled.squeeze(-1), labels_old, "ALT: Nullphasen enthalten")
        cluster_labels = pd.Series(labels_old, index=df_num.index)

    # -----------------------------
    # Export
    # -----------------------------
    export_df = pd.DataFrame({
        "Cluster": cluster_labels
    }, index=df_num.index).reset_index()
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Cluster-Zuordnung als CSV herunterladen", csv, f"Cluster_{mode.split()[0]}.csv", "text/csv")
