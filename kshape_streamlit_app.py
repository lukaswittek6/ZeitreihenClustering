import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape

st.set_page_config(page_title="Zeitreihen-Clustering", layout="centered")
st.title("Clustering für RELEX")
st.caption("20%-Regel wird vor Interpolation geprüft. Sondercluster wird mitgeplottet & in Plausibilitätsprüfung einbezogen.")

# ---------------- Params ----------------
EPS = 1e-9           # Toleranz für „quasi 0“
THRESH_SHARE = 0.20  # 20%-Schwelle
RANDOM_STATE = 42

# --------- UI: k-Auswahl ----------
cluster_option = st.radio("Clusteranzahl bestimmen:", ["Automatisch (Elbow)", "Manuell wählen"])
user_defined_k = st.slider("Wähle Clusteranzahl (k)", 2, 12, 4) if cluster_option == "Manuell wählen" else None

# --------- Upload ----------
uploaded_file = st.file_uploader("CSV/Excel hochladen", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # --- Laden ---
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # erste Spalte = ID
    df_raw.rename(columns={df_raw.columns[0]: "Markt"}, inplace=True)
    df_raw["Markt"] = df_raw["Markt"].astype(str).str.strip()
    df_raw.set_index("Markt", inplace=True)

    # Nur numerisch, alles andere -> NaN
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")

    # ---------- 20%-Regel VOR Interpolation ----------
    # „echte“ Datenpunkte = nicht NaN und |x| > EPS
    valid_mask = ~(df_num.isna() | (df_num.abs() <= EPS))
    data_ratio = valid_mask.sum(axis=1) / df_num.shape[1]
    sonder_mask = data_ratio <= THRESH_SHARE
    sonder_idx = df_num.index[sonder_mask]

    # ---------- erst jetzt Interpolation + NaN->0 für weitere Schritte ----------
    df_interp = df_num.interpolate(axis=1, limit_direction="both").fillna(0.0)

    # Split
    df_cluster = df_interp.drop(index=sonder_idx)                  # normal clustern
    df_sonder  = df_interp.loc[sonder_idx] if len(sonder_idx) else pd.DataFrame(columns=df_interp.columns)

    st.success(f"{len(df_interp)} Märkte geladen – {len(sonder_idx)} im Sondercluster (≤20% echte Punkte).")

    # Ergebnis-DF vorbereiten
    df_result = pd.DataFrame(index=df_interp.index)
    df_result["Cluster"] = "Sondercluster"  # default
    df_result["Plausible_Cluster"] = "Sondercluster"

    if df_cluster.empty:
        st.warning("Alle Märkte im Sondercluster – kein Clustering möglich.")
        # Trotzdem Sondercluster plotten (Original + z)
        if not df_sonder.empty:
            # Z-Scaling auf Sondercluster fitten (nur zur Visualisierung)
            scaler_sc = TimeSeriesScalerMeanVariance()
            X_sc = scaler_sc.fit_transform(df_sonder.values)

            fig, ax = plt.subplots(figsize=(10, 4))
            for serie in X_sc:
                ax.plot(range(X_sc.shape[1]), serie.flatten(), alpha=0.18)
            ax.plot(range(X_sc.shape[1]), X_sc.mean(axis=0).flatten(), color="black", lw=2, label="Mittel")
            ax.set_title("Sondercluster (z-transformiert)")
            ax.set_xlabel("Kalenderwoche (Index)"); ax.set_ylabel("Z-transformierter Absatz")
            ax.grid(True); ax.legend(); st.pyplot(fig)

            zeitspalten = df_interp.columns.tolist()
            fig, ax = plt.subplots(figsize=(12, 5))
            for _, row in df_sonder.iterrows():
                ax.plot(zeitspalten, row.values, alpha=0.25)
            if len(df_sonder) > 1:
                ax.plot(zeitspalten, df_sonder.mean().values, color="black", lw=2, label="Mittel")
            ax.set_title("Sondercluster – Absatzverläufe (Originaldaten)")
            ax.set_xlabel("Kalenderwoche"); ax.set_ylabel("Absatzmenge")
            ax.set_xticklabels(zeitspalten, rotation=45); ax.grid(True); ax.legend()
            st.pyplot(fig)

        # Export
        out = df_result.reset_index()
        st.download_button("📥 Cluster-Zuordnung herunterladen", out.to_csv(index=False).encode("utf-8"),
                           "Cluster_Zuordnung.csv", "text/csv")
        st.stop()

    # --------- Z-Scaling auf CLUSTER-Daten fitten ----------
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(df_cluster.values)

    # z-transformiertes Sondercluster (für Plots) mit gleichem Scaler
    X_sonder_scaled = scaler.transform(df_sonder.values) if not df_sonder.empty else None

    # ---------- k bestimmen (Elbow auf bereinigten Daten) ----------
    if user_defined_k is None:
        inertia, K_range = [], range(2, min(12, max(3, len(df_cluster))))
        for k in K_range:
            ks = KShape(n_clusters=k, random_state=RANDOM_STATE)
            ks.fit(X_scaled)
            inertia.append(ks.inertia_)
        diffs = np.diff(inertia)
        best_k_index = int(np.argmax(-diffs)) + 1
        optimal_k = list(K_range)[best_k_index]
        st.success(f"Automatisch gewählte Clusteranzahl: {optimal_k}")

        fig, ax = plt.subplots()
        ax.plot(list(K_range), inertia, marker="o")
        ax.set_title("Elbow-Methode (bereinigte Daten)")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia")
        st.pyplot(fig)
    else:
        optimal_k = int(user_defined_k)
        st.success(f"Manuell gewählte Clusteranzahl: {optimal_k}")

    # ---------- k-Shape auf df_cluster ----------
    kshape = KShape(n_clusters=optimal_k, random_state=RANDOM_STATE)
    labels = kshape.fit_predict(X_scaled)

    # Ergebnisse eintragen
    df_result.loc[df_cluster.index, "Cluster"] = labels
    df_result.loc[df_cluster.index, "Plausible_Cluster"] = labels  # ggf. später -1 setzen

    # ---------- Plots: z-transformiert inkl. Sondercluster ----------
    st.subheader("Clusterplots (z-transformiert, inkl. Sondercluster)")
    t_idx = list(range(X_scaled.shape[1]))

    # Normale Cluster
    for cid in sorted(set(labels)):
        fig, ax = plt.subplots(figsize=(10, 4))
        idxs = np.where(labels == cid)[0]
        block = X_scaled[idxs]
        for serie in block:
            ax.plot(t_idx, serie.flatten(), alpha=0.18)
        ax.plot(t_idx, block.mean(axis=0).flatten(), color="black", lw=2, label="Cluster-Mittel")
        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel("Kalenderwoche (Index)"); ax.set_ylabel("Z-transformierter Absatz")
        ax.grid(True); ax.legend(); st.pyplot(fig)

    # Sondercluster z-transformiert
    if X_sonder_scaled is not None and len(X_sonder_scaled) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        for serie in X_sonder_scaled:
            ax.plot(t_idx, serie.flatten(), alpha=0.18)
        ax.plot(t_idx, X_sonder_scaled.mean(axis=0).flatten(), color="black", lw=2, label="Mittel")
        ax.set_title("Sondercluster (z-transformiert)")
        ax.set_xlabel("Kalenderwoche (Index)"); ax.set_ylabel("Z-transformierter Absatz")
        ax.grid(True); ax.legend(); st.pyplot(fig)

    # ---------- 📌 Plausibilitätsprüfung ----------
    st.subheader("Plausibilitätsprüfung per Abweichung (inkl. Sondercluster)")
    delta_schwelle = st.slider("Wähle den Schwellenwert für die Abweichung Δ", 1.0, 3.0, 2.0, 0.1)

    abweichungen = []

    # normale Reihen: Distanz zum eigenen Zentrum
    centers = kshape.cluster_centers_
    for i, name in enumerate(df_cluster.index):
        lab = int(df_result.loc[name, "Cluster"])
        dist = float(np.sqrt(np.nanmean((X_scaled[i] - centers[lab]) ** 2)))
        abweichungen.append((name, lab, dist))

    # Sondercluster: Distanz zum nächstgelegenen Zentrum (nur Info)
    if X_sonder_scaled is not None and len(X_sonder_scaled) > 0:
        for i, name in enumerate(df_sonder.index):
            dists = [float(np.sqrt(np.nanmean((X_sonder_scaled[i] - c) ** 2))) for c in centers]
            abweichungen.append((name, "Sondercluster", float(min(dists))))

    ausreisser = [(n, l, round(d, 3)) for (n, l, d) in abweichungen if d > delta_schwelle]
    st.markdown(f"**{len(ausreisser)} Serien überschreiten Δ > {delta_schwelle}**")
    st.table(pd.DataFrame(abweichungen, columns=["Markt", "Cluster", "Abweichung"]).sort_values("Abweichung", ascending=False))

    # Plausible_Cluster: normale Reihen → -1 wenn über Schwelle; Sondercluster bleibt so
    dist_map = {n: d for (n, _, d) in abweichungen}
    for name in df_cluster.index:
        if dist_map.get(name, 0) > delta_schwelle:
            df_result.loc[name, "Plausible_Cluster"] = -1

    # ---------- 📈 Originalwerte: pro Cluster inkl. Sondercluster ----------
    st.subheader("Verlauf pro Cluster (Originaldaten, inkl. Sondercluster)")
    zeitspalten = df_interp.columns.tolist()

    for cid in sorted({c for c in df_result["Cluster"].unique() if isinstance(c, (int, np.integer))}):
        part = df_interp.loc[df_result["Cluster"] == cid]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in part.iterrows():
            ax.plot(zeitspalten, row.values, alpha=0.25)
        ax.plot(zeitspalten, part.mean().values, color="black", lw=2, label="Mittelwert")
        ax.set_title(f"Cluster {cid} – Absatzverläufe (Originaldaten)")
        ax.set_xlabel("Kalenderwoche"); ax.set_ylabel("Absatzmenge")
        ax.set_xticklabels(zeitspalten, rotation=45); ax.grid(True); ax.legend()
        st.pyplot(fig)

    sc_part = df_interp.loc[df_result["Cluster"] == "Sondercluster"]
    if not sc_part.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in sc_part.iterrows():
            ax.plot(zeitspalten, row.values, alpha=0.25)
        if len(sc_part) > 1:
            ax.plot(zeitspalten, sc_part.mean().values, color="black", lw=2, label="Mittelwert")
        ax.set_title("Sondercluster – Absatzverläufe (Originaldaten)")
        ax.set_xlabel("Kalenderwoche"); ax.set_ylabel("Absatzmenge")
        ax.set_xticklabels(zeitspalten, rotation=45); ax.grid(True); ax.legend()
        st.pyplot(fig)

    # ---------- Export ----------
    out = df_result.reset_index()
    st.download_button("📥 Cluster-Zuordnung (inkl. Sondercluster) herunterladen",
                       out.to_csv(index=False).encode("utf-8"),
                       "Cluster_Zuordnung.csv", "text/csv")
