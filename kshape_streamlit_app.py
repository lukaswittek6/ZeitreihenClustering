import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape

st.set_page_config(page_title="Zeitreihen-Clustering", layout="centered")

st.title("Clustering für RELEX")
st.markdown("Testversion mit Sondercluster-Regel (>20 % Nullen)")

# Clusterwahl
cluster_option = st.radio("Clusteranzahl bestimmen:", ["Automatisch (Elbow)", "Manuell wählen"])
if cluster_option == "Manuell wählen":
    user_defined_k = st.slider("Wähle Clusteranzahl (k)", 2, 10, 3)
else:
    user_defined_k = None

# Upload
uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx"])

if uploaded_file:
    # Excel laden
    df = pd.read_excel(uploaded_file)
    df.rename(columns={df.columns[0]: "Markt"}, inplace=True)
    df["Markt"] = df["Markt"].astype(str).str.strip()
    df.set_index("Markt", inplace=True)

    # NaNs auffüllen
    df = df.interpolate(axis=1, limit_direction="both").fillna(0)

    # Sondercluster-Regel: >20% Nullen
    zero_ratio = (df == 0).sum(axis=1) / df.shape[1]
    sondercluster_idx = zero_ratio[zero_ratio > 0.2].index

    # DataFrame für Clustering ohne Sondercluster
    df_cluster = df.drop(index=sondercluster_idx)

    st.success(f"{len(df)} Märkte geladen, davon {len(sondercluster_idx)} im Sondercluster.")

    if not df_cluster.empty:
        # Clustering
        X = df_cluster.values
        scaler = TimeSeriesScalerMeanVariance()
        X_scaled = scaler.fit_transform(X)

        # Elbow oder manuell
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

        kshape = KShape(n_clusters=optimal_k, random_state=42)
        labels = kshape.fit_predict(X_scaled)

        # Cluster-Ergebnisse zuordnen
        df_result = pd.DataFrame(index=df.index)
        df_result["Cluster"] = "Sondercluster"
        df_result.loc[df_cluster.index, "Cluster"] = labels

        # Plots pro Cluster (nur echte Cluster, nicht Sondercluster)
        zeitachsen = [f"{i+1}" for i in range(X_scaled.shape[1])]
        for cluster_id in sorted(set(labels)):
            fig, ax = plt.subplots(figsize=(10, 4))
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_data = X_scaled[cluster_indices]
            for serie in cluster_data:
                ax.plot(zeitachsen, serie.flatten(), alpha=0.2)
            mean_curve = cluster_data.mean(axis=0).flatten()
            ax.plot(zeitachsen, mean_curve, color="black", linewidth=2, label="Cluster-Mittel")
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xlabel("Kalenderwoche")
            ax.set_ylabel("Z-transformierter Absatz")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Export
        csv = df_result.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("📥 Cluster-Zuordnung als CSV herunterladen", csv, "Cluster_Zuordnung.csv", "text/csv")

    else:
        st.warning("Alle Märkte wurden dem Sondercluster zugeordnet – kein Clustering durchgeführt.")
