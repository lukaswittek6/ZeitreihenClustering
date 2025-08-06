
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape

# Streamlit UI
st.title("Zeitreihen-Clustering mit k-Shape")
st.markdown("Lade deine Excel-Datei mit Zeitreihen hoch (Marktspalte + Wochen), wähle die Clusteranzahl – automatisch oder manuell – und sieh dir das Ergebnis an.")

# Clusterwahl
cluster_option = st.radio("Clusteranzahl bestimmen:", ["Automatisch (Elbow)", "Manuell wählen"])
if cluster_option == "Manuell wählen":
    user_defined_k = st.slider("Wähle Clusteranzahl (k)", 2, 10, 3)
else:
    user_defined_k = None

# Upload
uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx"])

if uploaded_file:
    # Einlesen
    df = pd.read_excel(uploaded_file)
    df.rename(columns={df.columns[0]: "Markt"}, inplace=True)
    df["Markt"] = df["Markt"].astype(str).str.strip()
    df.set_index("Markt", inplace=True)

    # Vorbereitung
    df = df.interpolate(axis=1, limit_direction="both")
    df = df.dropna()
    st.success(f"{len(df)} Märkte erfolgreich geladen und bereinigt.")

    X = df.values
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)

    # Elbow-Methode oder manueller k
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

        # Plot anzeigen
        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker='o')
        ax.set_title("Elbow-Methode")
        ax.set_xlabel("Clusteranzahl (k)")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)
    else:
        optimal_k = user_defined_k
        st.success(f"Manuell gewählte Clusteranzahl: {optimal_k}")

    # Clustering
    kshape = KShape(n_clusters=optimal_k, random_state=42)
    labels = kshape.fit_predict(X_scaled)
    df['Cluster'] = labels

    # Ergebnis anzeigen
    st.subheader("📋 Cluster-Zuordnung")
    st.dataframe(df[['Cluster']])

    # Plot: Mittelwert je Cluster
    zeitachsen = [f"Woche {i+1}" for i in range(X_scaled.shape[1])]
    for cluster_id in np.unique(labels):
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

    # Download als CSV
    csv = df.reset_index()[["Markt", "Cluster"]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Cluster-Zuordnung als CSV herunterladen", csv, "Cluster_Zuordnung.csv", "text/csv")
