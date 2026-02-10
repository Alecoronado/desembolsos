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
YEAR_PREFIX = "Año"

FEATURE_COLS = [
    "Pais",
    "Sector",
    "SubSector",
    "TipodePrestamo",
    "Categoria Desembolso",
    "Monto_Total_Prestamo_Millones",
]

CAT_COLS = ["Pais", "Sector", "SubSector", "TipodePrestamo", "Categoria Desembolso"]
NUM_COLS = ["Monto_Total_Prestamo_Millones"]

# -----------------------------
# Helpers
# -----------------------------
def build_increment_targets_from_cum(Y_cum: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas acumuladas C_t en incrementos p_t:
      p_0 = C_0
      p_t = C_t - C_{t-1}
    Luego clip negativos y normaliza por fila para que sum(p)=1.
    """
    Y_cum = Y_cum.copy()

    # Ordenar columnas por número de año
    def year_key(col):
        import re
        m = re.findall(r"\d+", str(col))
        return int(m[-1]) if m else 0

    Y_cum = Y_cum[sorted(Y_cum.columns, key=year_key)]

    # Diferencias horizontales
    Y_inc = Y_cum.diff(axis=1)
    # Primer año: p0 = C0
    Y_inc.iloc[:, 0] = Y_cum.iloc[:, 0]

    # Limpiar: clip negativos
    Y_inc = Y_inc.clip(lower=0)

    # Normalizar para asegurar suma 1
    row_sum = Y_inc.sum(axis=1).replace(0, np.nan)
    Y_inc = Y_inc.div(row_sum, axis=0).fillna(0)

    return Y_inc

def prepare_XY(df: pd.DataFrame):
    # Validaciones
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas X: {missing}")

    Y_cols = [c for c in df.columns if str(c).startswith(YEAR_PREFIX)]
    if not Y_cols:
        raise ValueError(f"No se encontraron columnas objetivo con prefijo '{YEAR_PREFIX}'.")

    X = df[FEATURE_COLS].copy()
    Y_cum = df[Y_cols].copy()

    Y_inc = build_increment_targets_from_cum(Y_cum)

    p_cols = [f"p_{c}" for c in Y_inc.columns]
    Y_inc.columns = p_cols

    return X, Y_inc, Y_cum.columns.tolist(), p_cols

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
        "Categoria Desembolso": sorted(df["Categoria Desembolso"].dropna().unique())
    }
    sector_subsector_map = (
        df.groupby("Sector")["SubSector"]
        .unique()
        .apply(lambda x: sorted([v for v in x if pd.notna(v)]))
        .to_dict()
    )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[CAT_COLS])

    X_num_raw = np.log1p(X[NUM_COLS].astype(float))
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num_raw)

    X_proc = np.hstack([X_cat, X_num]).astype(np.float32)
    Y_np = Y_inc.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, Y_np, test_size=0.2, random_state=42
    )

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
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
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=32,
        callbacks=[cb],
        verbose=0
    )

    return model, encoder, scaler, year_cols, p_cols, unique_values, sector_subsector_map

model, encoder, scaler, year_cols, p_cols, unique_values, sector_subsector_map = load_and_train_model()

# -----------------------------
# Predict
# -----------------------------
def predict_distribution_and_cum(new_df: pd.DataFrame):
    X_cat = encoder.transform(new_df[CAT_COLS])
    X_num_raw = np.log1p(new_df[NUM_COLS].astype(float))
    X_num = scaler.transform(X_num_raw)

    X_proc = np.hstack([X_cat, X_num]).astype(np.float32)

    p_pred = model.predict(X_proc, verbose=0)
    p_pred = np.clip(p_pred, 0, 1)

    C_pred = np.cumsum(p_pred, axis=1)
    C_pred[:, -1] = 1.0

    df_p = pd.DataFrame(p_pred, columns=p_cols)
    df_C = pd.DataFrame(C_pred, columns=year_cols)

    return df_p, df_C

# -----------------------------
# App UI
# -----------------------------
st.title("📊 Predicción de Curvas de Desembolso (Softmax → siempre llega a 1)")

st.write("El modelo predice el **desembolso anual** (incrementos) y luego construye el acumulado con cumsum.")

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
    monto = st.number_input("💵 Monto Total del Préstamo (Millones)", value=40.0, min_value=0.0)

if st.button("Predecir Curva"):
    new_df = pd.DataFrame({
        "Pais": [pais],
        "Sector": [sector],
        "SubSector": [subsector],
        "TipodePrestamo": [tipodeprestamo],
        "Categoria Desembolso": [categoria],
        "Monto_Total_Prestamo_Millones": [monto],
    })

    df_p, df_C = predict_distribution_and_cum(new_df)

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
