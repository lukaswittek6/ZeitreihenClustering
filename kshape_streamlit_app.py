import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape

st.set_page_config(page_title="Zeitreihen-Clustering", layout="centered")

st.title("Clustering für RELEX")
st.markdown("Testversion.")

# Clusterwahl
cluster_option = st.radio("Clusteranzahl bestimmen:", ["Automatisch (Elbow)", "Manuell wählen"])
if cluster_option == "Manuell wählen":
    user_defined_k = st.slider("Wähle Clusteranzahl (k)", 2, 10, 3)
else:
    user_defined_k = None

# Upload
uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx"])

if uploaded_file:
    # ===== Laden & vorbereiten =====
    df_raw = pd.read_excel(uploaded_file)
    df_raw.rename(columns={df_raw.columns[0]: "Markt"}, inplace=True)
    df_raw["Markt"] = df_raw["Markt"].astype(str).str.strip()
    df_raw.set_index("Markt", inplace=True)

    # Zahlen erzwingen
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")

    # ===== Slider für 20%-Regel =====
    min_data_ratio = st.slider(
        "Mindestanteil an echten Daten zur Datenbereinigung",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05
    )

    # ===== Sondercluster-Logik vor Interpolation =====
    EPS = 1e-9
    valid_mask = ~(df_num.isna() | (df_num.abs() <= EPS))
    data_ratio = valid_mask.sum(axis=1) / df_num.shape[1]
    sonder_mask = data_ratio <= min_data_ratio
    sonder_idx = df_num.index[sonder_mask]

    # ===== Interpolation für Clustering =====
    df_interp = df_num.interpolate(axis=1, limit_direction="both").fillna(0.0)

    # ===== Clustern nur für Nicht-Sondercluster =====
    df_cluster = df_interp.drop(index=sonder_idx)
    X = df_cluster.values
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)

    if user_defined_k is None:
        inertia = []
        K_range = range(2, 10)
        for k in K_range:
            ks = KShape(n_clusters=k, random_state=42)
            ks.fit(X_scaled)
            inertia.append(ks.inertia_)

        diffs = np.diff(inertia)
        best_k_index = np.argmax(-diffs) + 1
        optimal_k = K_range[best_k_index]
        st.success(f"Automatisch gewählte Clusteranzahl: {optimal_k}")

        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker='o')
        ax.set_title("Elbow-Methode")
        ax.set_xlabel("Clusteranzahl (k)")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)
    else:
        optimal_k = user_defined_k
        st.success(f"Manuell gewählte Clusteranzahl: {optimal_k}")

    # ===== KShape-Clustering =====
    kshape = KShape(n_clusters=optimal_k, random_state=42)
    labels = kshape.fit_predict(X_scaled)

    # ===== Ergebnis-DF zusammenbauen =====
    df_result = pd.DataFrame(index=df_interp.index)
    df_result["Cluster"] = "Sondercluster"
    df_result.loc[df_cluster.index, "Cluster"] = labels

    # ===== Plots pro Cluster (inkl. Sondercluster) – z-transformiert =====
    zeitachsen = [f"{i+1}" for i in range(df_interp.shape[1])]

    for cluster_id in sorted(df_result["Cluster"].unique(), key=lambda x: (isinstance(x, int), x)):
        fig, ax = plt.subplots(figsize=(10, 4))
        idxs = df_result[df_result["Cluster"] == cluster_id].index
        data = df_interp.loc[idxs]
        scaled_data = scaler.fit_transform(data.values)
        for serie in scaled_data:
            ax.plot(zeitachsen, serie.flatten(), alpha=0.2)
        mean_curve = scaled_data.mean(axis=0).flatten()
        ax.plot(zeitachsen, mean_curve, color="black", linewidth=2, label="Cluster-Mittel")
        ax.set_title(f"Cluster {cluster_id} ({len(idxs)} Märkte)")
        ax.set_xlabel("Kalenderwoche")
        ax.set_ylabel("Z-transformierter Absatz")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Sondercluster bekommt automatisch plausible = "Sondercluster"
    df_result.loc[sonder_idx, "Plausible_Cluster"] = "Sondercluster"

    # ===== Plots pro Cluster (Originaldaten) =====
    st.subheader("Verlauf pro Cluster (Originaldaten)")
    zeitspalten = df_interp.columns

    for cluster_id in sorted(df_result["Cluster"].unique(), key=lambda x: (isinstance(x, int), x)):
        cluster_df = df_interp.loc[df_result["Cluster"] == cluster_id]
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in cluster_df.iterrows():
            ax.plot(zeitspalten, row, alpha=0.3)
        mean_curve = cluster_df.mean()
        ax.plot(zeitspalten, mean_curve, color="black", linewidth=2, label="Cluster-Mittelwert")
        ax.set_title(f"Cluster {cluster_id} – Absatzverläufe (Originaldaten)")
        ax.set_xlabel("Kalenderwoche")
        ax.set_ylabel("Absatzmenge")
        ax.set_xticklabels(zeitspalten, rotation=45)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    # ===== Export =====
    csv = df_result.reset_index()[["Markt", "Cluster", "Plausible_Cluster"]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Cluster-Zuordnung als CSV herunterladen", csv, "Cluster_Zuordnung.csv", "text/csv")
