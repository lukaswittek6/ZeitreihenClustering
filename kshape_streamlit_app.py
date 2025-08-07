# 📦 Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# 🎨 Streamlit Layout
st.set_page_config(page_title="K-Means Zeitreihen-Clustering", layout="centered")
st.title("K-Means Clustering von Absatzzeitreihen")

# 📁 Datei-Upload
uploaded_file = st.file_uploader("⬆️ Lade deine Excel-Datei mit Absatzdaten hoch", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, header=0)
    df.rename(columns={df.columns[0]: "Markt"}, inplace=True)
    df["Markt"] = df["Markt"].astype(str).str.strip()
    df.set_index("Markt", inplace=True)

    # 📌 Interpolation und Skalierung
    df = df.interpolate(axis=1, limit_direction="both")
    df = df.dropna()
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 📈 Elbow-Methode zur Clusterwahl
    st.subheader("📐 Clusteranzahl bestimmen (Elbow-Methode)")
    K_range = range(1, 11)
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, inertias, marker='o')
    ax.set_title("Elbow-Methode – Clusteranzahlbestimmung")
    ax.set_xlabel("Anzahl Cluster (k)")
    ax.set_ylabel("Trägheit (Inertia)")
    st.pyplot(fig)

    # 👤 Auswahl der Clusteranzahl
    user_k = st.slider("Wähle die gewünschte Clusteranzahl (k)", min_value=2, max_value=10, value=3)
    km = KMeans(n_clusters=user_k, random_state=42, n_init='auto')
    labels = km.fit_predict(X_scaled)

    df["Cluster"] = labels

    # 📊 Plot pro Cluster mit Mittelwert
    st.subheader("📉 Visualisierung der Cluster")
    wochen_labels = [f"Woche {i+1}" for i in range(X.shape[1])]

    for cluster_id in np.unique(labels):
        cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1).values

        fig, ax = plt.subplots(figsize=(14, 5))
        for serie in cluster_data:
            ax.plot(wochen_labels, serie, alpha=0.2)

        cluster_mean = cluster_data.mean(axis=0)
        ax.plot(wochen_labels, cluster_mean, color="black", linewidth=2, label="Cluster-Mittelwert")
        ax.set_title(f"Cluster {cluster_id} – Absatzverläufe")
        ax.set_xlabel("Kalenderwoche")
        ax.set_ylabel("Absatz (standardisiert)")
        ax.set_xticks(range(len(wochen_labels)))
        ax.set_xticklabels(wochen_labels, rotation=45)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)

    # 📤 CSV Export der Cluster-Zuordnung
    st.subheader("📥 Export der Cluster-Zuordnung")
    df_export = df[["Cluster"]].reset_index()
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Cluster-Zuordnung als CSV herunterladen",
        data=csv,
        file_name="cluster_zuordnung_kmeans.csv",
        mime="text/csv"
    )
