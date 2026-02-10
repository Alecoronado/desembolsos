import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# Config
# -----------------------------
FILE_PATH = "Desembolsos_Softmax.xlsx"
SHEET_NAME = "Sheet1"
YEAR_PREFIX = "p_Año"  # columnas tipo "p_Año0", "p_Año1", ...

FEATURE_COLS = [
    "Pais",
    "Sector",
    "SubSector",
    "TipodePrestamo",
    "Categoria Desembolso",
    "Monto Prestamo",
]

CAT_COLS = ["Pais", "Sector", "SubSector", "TipodePrestamo", "Categoria Desembolso"]
NUM_COLS = ["Monto Prestamo"]

# -----------------------------
# Helpers
# -----------------------------
def prepare_XY(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas X: {missing}")

    p_cols = [c for c in df.columns if str(c).startswith(YEAR_PREFIX)]
    if not p_cols:
        raise ValueError(f"No se encontraron columnas objetivo con prefijo '{YEAR_PREFIX}'.")

    def year_key(col):
        import re
        m = re.findall(r"\d+", str(col))
        return int(m[-1]) if m else 0

    p_cols = sorted(p_cols, key=year_key)

    X = df[FEATURE_COLS].copy()
    Y_inc = df[p_cols].copy().astype(float)

    row_sum = Y_inc.sum(axis=1).replace(0, np.nan)
    Y_inc = Y_inc.div(row_sum, axis=0).fillna(0)

    year_cols = [c.replace("p_", "") for c in p_cols]
    return X, Y_inc, year_cols, p_cols

def compute_t95_from_p(P: np.ndarray) -> np.ndarray:
    C = np.cumsum(P, axis=1)
    return (C >= 0.95).argmax(axis=1)

def plot_curve_from_cum(C: np.ndarray, year_cols, start_date, title):
    dates = [start_date + timedelta(days=365 * i) for i in range(len(year_cols))]
    plt.figure(figsize=(8, 5))
    plt.plot(dates, C, marker="o", linestyle="-")
    for i, v in enumerate(C):
        plt.text(dates[i], v, f"{v:.2f}", ha="right", fontsize=9)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Porcentaje acumulado")
    return plt

def apply_temperature(p_pred: np.ndarray, tau: float) -> np.ndarray:
    """Ajuste de temperatura sobre probabilidades (aprox. softmax(z/tau))"""
    p = np.clip(p_pred, 1e-12, 1.0)
    p = np.power(p, 1.0 / tau)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def cum_from_p(p: np.ndarray) -> np.ndarray:
    C = np.cumsum(p, axis=1)
    C[:, -1] = 1.0
    return C

def tune_tau_by_category(model, X_proc_full, df_full, p_cols, categoria_col="Categoria Desembolso"):
    """
    Encuentra el mejor tau por categoría minimizando MAE del acumulado (C)
    entre curva real y curva predicha ajustada con tau.
    """
    P_true = df_full[p_cols].values.astype(np.float32)
    P_true = P_true / np.maximum(P_true.sum(axis=1, keepdims=True), 1e-12)
    C_true = cum_from_p(P_true)

    P_pred = model.predict(X_proc_full, verbose=0).astype(np.float32)

    taus = np.round(np.arange(0.60, 1.41, 0.05), 2)
    tau_map = {}

    for cat in sorted(df_full[categoria_col].dropna().unique()):
        idx = (df_full[categoria_col].values == cat)
        n = int(idx.sum())

        if n < 3:
            tau_map[cat] = {"tau": 1.0, "n": n, "mae": None}
            continue

        best_tau, best_score = 1.0, 1e18
        for tau in taus:
            P_adj = apply_temperature(P_pred[idx], float(tau))
            C_adj = cum_from_p(P_adj)
            score = float(np.mean(np.abs(C_true[idx] - C_adj)))

            if score < best_score:
                best_score = score
                best_tau = float(tau)

        tau_map[cat] = {"tau": best_tau, "n": n, "mae": float(best_score)}

    return tau_map

# -----------------------------
# Train with cache
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_and_train_model():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

    X, Y_inc, year_cols, p_cols = prepare_XY(df)

    unique_values = {
        "Pais": sorted(df["Pais"].dropna().unique()),
        "Sector": sorted(df["Sector"].dropna().unique()),
        "TipodePrestamo": sorted(df["TipodePrestamo"].dropna().unique()),
        "Categoria Desembolso": sorted(df["Categoria Desembolso"].dropna().unique()),
    }
    sector_subsector_map = (
        df.groupby("Sector")["SubSector"]
        .unique()
        .apply(lambda x: sorted([v for v in x if pd.notna(v)]))
        .to_dict()
    )

    # --- Preprocesamiento X ---
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[CAT_COLS])

    X_num_raw = np.log1p(X[NUM_COLS].astype(float))
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num_raw)

    X_proc = np.hstack([X_cat, X_num]).astype(np.float32)
    Y_np = Y_inc.values.astype(np.float32)

    # --- Balanceo por velocidad (t95) ---
    t95 = compute_t95_from_p(Y_np)
    speed_class = np.where(t95 <= 1, 0, np.where(t95 <= 4, 1, 2))

    counts = np.bincount(speed_class, minlength=3).astype(float)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    sample_weight = class_weights[speed_class].astype(np.float32)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_proc, Y_np, sample_weight, test_size=0.2, random_state=42
    )

    # --- Modelo más pequeño (pocos datos) ---
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(y_train.shape[1], activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )

    cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        validation_data=(X_test, y_test, w_test),
        epochs=400,
        batch_size=16,
        callbacks=[cb],
        verbose=0
    )

    # --- Calibrar tau por categoría (con TODO el dataset) ---
    df_full = pd.concat([X.reset_index(drop=True), Y_inc.reset_index(drop=True)], axis=1)

    X_cat_full = encoder.transform(X[CAT_COLS])
    X_num_raw_full = np.log1p(X[NUM_COLS].astype(float))
    X_num_full = scaler.transform(X_num_raw_full)
    X_proc_full = np.hstack([X_cat_full, X_num_full]).astype(np.float32)

    tau_map = tune_tau_by_category(
        model=model,
        X_proc_full=X_proc_full,
        df_full=df_full,
        p_cols=p_cols,
        categoria_col="Categoria Desembolso"
    )

    return model, encoder, scaler, year_cols, p_cols, unique_values, sector_subsector_map, tau_map

model, encoder, scaler, year_cols, p_cols, unique_values, sector_subsector_map, tau_map = load_and_train_model()

# -----------------------------
# Predict
# -----------------------------
def predict_distribution_and_cum(new_df: pd.DataFrame, tau: float = 1.0):
    X_cat = encoder.transform(new_df[CAT_COLS])
    X_num_raw = np.log1p(new_df[NUM_COLS].astype(float))
    X_num = scaler.transform(X_num_raw)

    X_proc = np.hstack([X_cat, X_num]).astype(np.float32)

    p_pred = model.predict(X_proc, verbose=0)

    if tau != 1.0:
        p_pred = apply_temperature(p_pred, tau)

    p_pred = np.clip(p_pred, 0, 1)

    C_pred = np.cumsum(p_pred, axis=1)
    C_pred[:, -1] = 1.0

    df_p = pd.DataFrame(p_pred, columns=p_cols)
    df_C = pd.DataFrame(C_pred, columns=year_cols)
    return df_p, df_C

# -----------------------------
# App UI
# -----------------------------
st.title("📊 Predicción de Curvas de Desembolso (τ automático por categoría)")
st.write("El modelo predice **incrementos (p)** con softmax y construye el acumulado con cumsum (cierra en 1).")

nombre_proyecto = st.text_input("🏗️ Nombre del Proyecto:")
fecha_inicio = st.date_input("📅 Fecha de Inicio del Proyecto:", datetime.today())

col1, col2 = st.columns(2)

with col1:
    pais = st.selectbox("🌍 País:", options=unique_values["Pais"])
    sector = st.selectbox("🏭 Sector:", options=unique_values["Sector"])
    subsectores = sector_subsector_map.get(sector, [])
    subsector = st.selectbox("🏢 SubSector:", options=subsectores)

with col2:
    tipodeprestamo = st.selectbox("💰 Tipo de Préstamo:", options=unique_values["TipodePrestamo"])
    categoria = st.selectbox("📊 Categoría de Desembolso:", options=unique_values["Categoria Desembolso"])
    monto = st.number_input("💵 Monto del Préstamo (Millones)", value=40.0, min_value=0.0)

# Mostrar tau recomendado para la categoría seleccionada
tau_auto = tau_map.get(categoria, {}).get("tau", 1.0)
n_cat = tau_map.get(categoria, {}).get("n", 0)
st.caption(f"τ automático para '{categoria}': **{tau_auto:.2f}** (n={n_cat})")

if st.button("Predecir Curva"):
    new_df = pd.DataFrame({
        "Pais": [pais],
        "Sector": [sector],
        "SubSector": [subsector],
        "TipodePrestamo": [tipodeprestamo],
        "Categoria Desembolso": [categoria],
        "Monto Prestamo": [monto],
    })

    df_p, df_C = predict_distribution_and_cum(new_df, tau=tau_auto)

    st.subheader("🧾 Desembolso anual predicho (incrementos, suma=1)")
    st.dataframe(df_p.style.format("{:.4f}"))

    st.subheader("📈 Curva acumulada predicha (siempre cierra en 1)")
    st.dataframe(df_C.style.format("{:.4f}"))

    st.subheader("📉 Gráfico de la curva acumulada")
    plt_obj = plot_curve_from_cum(
        df_C.iloc[0].values,
        year_cols=year_cols,
        start_date=fecha_inicio,
        title=f"Curva de Desembolso - {nombre_proyecto or 'Nuevo Proyecto'}"
    )
    st.pyplot(plt_obj.gcf())
