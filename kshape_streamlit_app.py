import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape

st.set_page_config(page_title="Zeitreihen-Clustering", layout="centered")

st.title("Clustering für RELEX")
st.caption("Sondercluster-Regel: >20 % quasi Null (NaN oder |x| ≤ EPS). Sondercluster wird mitgeplottet & in die Plausibilitätsprüfung einbezogen.")

# ---------------- Params ----------------
EPS = 1e-9
NULL_SHARE = 0.20
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

    df_raw.rename(columns={df_raw.columns[0]: "Markt"}, inplace=True)
    df_raw["Markt"] = df_raw["Markt"].astype(str).str.strip()
    df_raw.set_index("Markt", inplace=True)

    # --- numerisch + Interpolation entlang Zeitachse ---
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    df_interp = df_num.interpolate(axis=1, limit_direction="both")

    # --- Sondercluster-Regel: >20% quasi 0 (NaN ODER |x| ≤ EPS) ---
    is_quasi_zero = df_interp.isna() | (df_interp.abs() <= EPS)
    zero_ratio = is_quasi_zero.sum(axis=1) / df_interp.shape[1]
    sonder_idx = zero_ratio[zero_ratio > NULL_SHARE].index

    st.success(f"{len(df_interp)} Märkte geladen – {len(sonder_idx)} im Sondercluster (>20% quasi 0).")

    # Fürs Clustering NaNs -> 0
    df_filled = df_interp.fillna(0.0)

    # Datensätze splitten
    df_cluster = df_filled.drop(index=sonder_idx)           # normal zu clustern
    df_sonder  = df_filled.loc[sonder_idx] if len(sonder_idx) else pd.DataFrame(columns=df_filled.columns)

    if df_cluster.empty:
        st.warning("Alle Märkte im Sondercluster – kein Clustering möglich.")
        # Ergebnis-DF nur mit Sondercluster
        df_result = pd.DataFrame(index=df_filled.index)
        df_result["Cluster"] = "Sondercluster"
    else:
        # --------- Z-Scaling nur auf den Clustering-Datensatz fitten ----------
        scaler = TimeSeriesScalerMeanVariance()
        X_scaled = scaler.fit_transform(df_cluster.values)

        # --- k bestimmen (Elbow auf bereinigten Daten) ---
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
            ax.set_xlabel("k"); ax.set_ylabel("Inertia"); st.pyplot(fig)
        else:
            optimal_k = int(user_defined_k)
            st.success(f"Manuell gewählte Clusteranzahl: {optimal_k}")

        # --------- k-Shape nur auf df_cluster ----------
        kshape = KShape(n_clusters=optimal_k, random_state=RANDOM_STATE)
        labels = kshape.fit_predict(X_scaled)

        # Ergebnis-Liste auf alle Märkte mappen
        df_result = pd.DataFrame(index=df_filled.index)
        df_result["Cluster"] = "Sondercluster"
        df_result.loc[df_cluster.index, "Cluster"] = labels

        # --------- Z-Transformierte Daten auch für Sondercluster ----------
        # Wir transformieren Sondercluster-Serien mit DEMSELBEN Scaler, damit sie in derselben Z-Skala liegen.
        X_sonder_scaled = None
        if not df_sonder.empty:
            X_sonder_scaled = scaler.transform(df_sonder.values)

        # --------- Plots: z-transformiert inkl. Sondercluster ----------
        st.subheader("Clusterplots (z-transformiert, inkl. Sondercluster)")

        # Normale Cluster
        zeitachsen_idx = list(range(X_scaled.shape[1]))
        for cid in sorted(set(labels)):
            fig, ax = plt.subplots(figsize=(10, 4))
            idxs = np.where(labels == cid)[0]
            block = X_scaled[idxs]
            for serie in block:
                ax.plot(zeitachsen_idx, serie.flatten(), alpha=0.18)
            ax.plot(zeitachsen_idx, block.mean(axis=0).flatten(), color="black", lw=2, label="Cluster-Mittel")
            ax.set_title(f"Cluster {cid}")
            ax.set_xlabel("Kalenderwoche (Index)"); ax.set_ylabel("Z-transformierter Absatz")
            ax.grid(True); ax.legend(); st.pyplot(fig)

        # Sondercluster z-transformiert (falls vorhanden)
        if X_sonder_scaled is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            for serie in X_sonder_scaled:
                ax.plot(zeitachsen_idx, serie.flatten(), alpha=0.18)
            # Mittelwert (falls mehrere Reihen)
            if len(X_sonder_scaled) > 0:
                ax.plot(zeitachsen_idx, X_sonder_scaled.mean(axis=0).flatten(), color="black", lw=2, label="Cluster-Mittel")
            ax.set_title("Sondercluster (z-transformiert)")
            ax.set_xlabel("Kalenderwoche (Index)"); ax.set_ylabel("Z-transformierter Absatz")
            ax.grid(True); ax.legend(); st.pyplot(fig)

        # --------- 📌 Plausibilitätsprüfung (inkl. Sondercluster) ----------
        st.subheader("Plausibilitätsprüfung per Abweichung (inkl. Sondercluster)")
        delta_schwelle = st.slider("Wähle den Schwellenwert für die Abweichung Δ", 1.0, 3.0, 2.0, 0.1)

        abweichungen = []

        # 1) normale Reihen: Distanz zu eigenem Clusterzentrum
        centers = kshape.cluster_centers_
        for i, name in enumerate(df_cluster.index):
            lab = int(df_result.loc[name, "Cluster"])
            dist = float(np.sqrt(np.nanmean((X_scaled[i] - centers[lab]) ** 2)))
            abweichungen.append((name, lab, dist))

        # 2) Sondercluster: Distanz zum NÄCHSTEN Zentrum (nur Info; Label bleibt "Sondercluster")
        if X_sonder_scaled is not None and len(X_sonder_scaled) > 0:
            for i, name in enumerate(df_sonder.index):
                dists_to_all = [float(np.sqrt(np.nanmean((X_sonder_scaled[i] - c) ** 2))) for c in centers]
                dist = min(dists_to_all)
                abweichungen.append((name, "Sondercluster", dist))

        ausreisser = [(n, l, round(d, 3)) for (n, l, d) in abweichungen if d > delta_schwelle]
        st.markdown(f"**{len(ausreisser)} Serien überschreiten Δ > {delta_schwelle}**")
        st.table(pd.DataFrame(abweichungen, columns=["Markt", "Cluster", "Abweichung"]).sort_values("Abweichung", ascending=False))

        # Plausible_Cluster-Spalte: normale Reihen -> -1 bei Ausreißern; Sondercluster bleibt "Sondercluster"
        dist_map = {n: d for (n, _, d) in abweichungen}
        df_result["Plausible_Cluster"] = df_result["Cluster"]
        for name in df_cluster.index:  # nur normale Reihen
            lab = df_result.loc[name, "Cluster"]
            if isinstance(lab, (int, np.integer)) and dist_map.get(name, 0) > delta_schwelle:
                df_result.loc[name, "Plausible_Cluster"] = -1

    # --------- 📈 Originalwerte: Verlauf pro Cluster inkl. Sondercluster ----------
    st.subheader("Verlauf pro Cluster (Originaldaten, inkl. Sondercluster)")
    zeitspalten = df_filled.columns.tolist()
    for cluster_id in sorted({c for c in df_result["Cluster"].unique() if isinstance(c, (int, np.integer))}):
        cluster_df = df_filled.loc[df_result["Cluster"] == cluster_id]
        if cluster_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in cluster_df.iterrows():
            ax.plot(zeitspalten, row.values, alpha=0.25)
        mean_curve = cluster_df.mean()
        ax.plot(zeitspalten, mean_curve.values, color="black", lw=2, label="Cluster-Mittelwert")
        ax.set_title(f"Cluster {cluster_id} – Absatzverläufe (Originaldaten)")
        ax.set_xlabel("Kalenderwoche"); ax.set_ylabel("Absatzmenge")
        ax.set_xticklabels(zeitspalten, rotation=45); ax.grid(True); ax.legend()
        st.pyplot(fig)

    # Sondercluster – Originalwerte
    sc_df = df_filled.loc[df_result["Cluster"] == "Sondercluster"]
    if not sc_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in sc_df.iterrows():
            ax.plot(zeitspalten, row.values, alpha=0.25)
        if len(sc_df) > 1:
            ax.plot(zeitspalten, sc_df.mean().values, color="black", lw=2, label="Cluster-Mittelwert")
        ax.set_title("Sondercluster – Absatzverläufe (Originaldaten)")
        ax.set_xlabel("Kalenderwoche"); ax.set_ylabel("Absatzmenge")
        ax.set_xticklabels(zeitspalten, rotation=45); ax.grid(True); ax.legend()
        st.pyplot(fig)

    # --------- Export ----------
    out = df_result.reset_index()
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Cluster-Zuordnung (inkl. Sondercluster) herunterladen", csv, "Cluster_Zuordnung.csv", "text/csv")
