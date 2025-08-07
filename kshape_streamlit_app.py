import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
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
    df = pd.read_excel(uploaded_file)
    df.rename(columns={df.columns[0]: "Markt"}, inplace=True)
    df["Markt"] = df["Markt"].astype(str).str.strip()
    df.set_index("Markt", inplace=True)
    df = df.interpolate(axis=1, limit_direction="both").dropna()

    st.success(f"{len(df)} Märkte erfolgreich geladen.")

    X = df.values
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

    kshape = KShape(n_clusters=optimal_k, random_state=42)
    labels = kshape.fit_predict(X_scaled)
    df['Cluster'] = labels

    zeitachsen = [f"{i+1}" for i in range(X_scaled.shape[1])]

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

    # 📌 Abschnitt: Delta-Slider zur Plausibilitätsprüfung
    st.subheader("Plausibilitätsprüfung per Abweichung")

    delta_schwelle = st.slider(
        "Wähle den Schwellenwert für die Abweichung Δ",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    abweichungen = []
    for idx, label in enumerate(labels):
        dist = np.sqrt(np.mean((X_scaled[idx] - kshape.cluster_centers_[label]) ** 2))
        abweichungen.append((df.index[idx], label, dist))

    ausreisser = [(name, label, round(dist, 2)) 
                  for name, label, dist in abweichungen if dist > delta_schwelle]

    st.markdown(f"**{len(ausreisser)} Märkte überschreiten Δ > {delta_schwelle}**")
    if ausreisser:
        st.table(pd.DataFrame(ausreisser, columns=["Markt", "Cluster", "Abweichung"]))
    else:
        st.success("Alle Märkte liegen innerhalb der akzeptierten Abweichung.")

    df["Plausible_Cluster"] = [
        -1 if dist > delta_schwelle else cluster
        for (_, cluster, dist) in abweichungen
    ]

    # 📈 Verlauf pro Cluster (Originalskala)
    st.subheader("Verlauf pro Cluster (Mittelwert & Einzelverläufe auf Originaldaten)")
    
    zeitspalten = df.columns[:-2]  # alle außer Cluster & Plausible_Cluster
    
    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_df = df[df["Cluster"] == cluster_id]
        fig, ax = plt.subplots(figsize=(12, 5))
    
        for idx, row in cluster_df.iterrows():
            ax.plot(zeitspalten, row[zeitspalten], alpha=0.3)
    
        mean_curve = cluster_df[zeitspalten].mean()
        ax.plot(zeitspalten, mean_curve, color="black", linewidth=2, label="Cluster-Mittelwert")
    
        ax.set_title(f"Cluster {cluster_id} – Absatzverläufe (Originaldaten)")
        ax.set_xlabel("Kalenderwoche")
        ax.set_ylabel("Absatzmenge")
        ax.set_xticklabels(zeitspalten, rotation=45)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    
    # Export
    csv = df.reset_index()[["Markt", "Cluster", "Plausible_Cluster"]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Cluster-Zuordnung als CSV herunterladen", csv, "Cluster_Zuordnung_KMeans.csv", "text/csv")
