
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# 📌 Seiteneinstellungen
st.set_page_config(page_title="Synchrones Clustering mit K-Means", layout="wide")

# 📌 Titel
st.title("🧠 Synchrones Zeitreihen-Clustering mit K-Means")
st.markdown("Diese App führt ein synchrones Clustering mit K-Means durch – zur Erkennung von ähnlichem Trendverhalten **zur gleichen Zeit**.")

# 📌 Datei-Upload
st.sidebar.header("📤 Excel-Datei hochladen")
uploaded_file = st.sidebar.file_uploader("Wähle eine Excel-Datei mit Absatzwerten", type=["xlsx"])

# 📌 Clusteranzahl manuell einstellbar
n_clusters = st.sidebar.slider("Anzahl der Cluster (k)", min_value=2, max_value=10, value=3)

# 📌 Hauptlogik
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.rename(columns={df.columns[0]: "Markt"}, inplace=True)
    df["Markt"] = df["Markt"].astype(str)
    df.set_index("Markt", inplace=True)

    # Interpolation & Skalierung
    df = df.interpolate(axis=1, limit_direction="both")
    df = df.dropna()
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 📌 Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = labels

    st.success(f"✅ Clustering erfolgreich mit {n_clusters} Clustern durchgeführt.")
    st.dataframe(df.head())

    # 📌 Visualisierung
    zeitachsen = df.columns[:-1]  # alle Spalten außer "Cluster"

    for cluster_id in np.unique(labels):
        plt.figure(figsize=(12, 4))
        cluster_df = df[df["Cluster"] == cluster_id]
        for row in cluster_df.iloc[:, :-1].values:
            plt.plot(zeitachsen, row, alpha=0.2, linewidth=1)

        # Mittelwert
        mean_curve = cluster_df.iloc[:, :-1].mean().values
        plt.plot(zeitachsen, mean_curve, color="black", linewidth=2, label="Cluster-Mittel")

        plt.title(f"Cluster {cluster_id}: Synchroner Verlauf")
        plt.xlabel("Kalenderwoche")
        plt.ylabel("Standardisierter Absatz")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

    # 📌 Cluster-Zuordnung exportieren
    st.markdown("### 📥 Cluster-Zuordnung als CSV exportieren")
    clustered_df = df[["Cluster"]].reset_index()
    csv = clustered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📁 CSV herunterladen", data=csv, file_name="Cluster_Zuordnung_KMeans.csv", mime="text/csv")

else:
    st.warning("⬅ Bitte lade eine Excel-Datei hoch, um zu starten.")
