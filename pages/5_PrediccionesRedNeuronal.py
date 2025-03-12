import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 📌 Cargar y entrenar el modelo con caché
@st.cache_resource(show_spinner=False)
def load_and_train_model():
    file_path = "Desembolsos_Acumulados.xlsx"
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 📌 Verificar columnas necesarias
    required_columns = ['Pais', 'Sector', 'SubSector', 'TipodePrestamo', 'Categoria Desembolso', 'Monto_Total_Prestamo_Millones']
    Y_columns = [col for col in df.columns if col.startswith('Año')]
    
    if not set(required_columns).issubset(df.columns) or not Y_columns:
        st.error("❌ El archivo `Desem.xlsx` no contiene las columnas esperadas.")
        return None, None, None, None, None

    df[Y_columns] = df[Y_columns].cumsum(axis=1).clip(upper=1)

    unique_values = {
        "Pais": sorted(df["Pais"].dropna().unique()),
        "Sector": sorted(df["Sector"].dropna().unique()),
        "TipodePrestamo": sorted(df["TipodePrestamo"].dropna().unique()),
        "Categoria Desembolso": sorted(df["Categoria Desembolso"].dropna().unique())
    }

    sector_subsector_map = df.groupby("Sector")["SubSector"].unique().apply(sorted).to_dict()

    X = df[required_columns]
    Y = df[Y_columns]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical = encoder.fit_transform(X[['Pais', 'Sector', 'SubSector', 'TipodePrestamo', 'Categoria Desembolso']])

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X[['Monto_Total_Prestamo_Millones']])

    X_processed = np.hstack([X_categorical, X_numeric])

    X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(Y_train.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.Huber(),
                  metrics=['mae'])
    model.fit(X_train, Y_train, epochs=300, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

    return model, encoder, scaler, Y_columns, unique_values, sector_subsector_map

model, encoder, scaler, Y_columns, unique_values, sector_subsector_map = load_and_train_model()

# 📌 Función para predecir la curva de desembolso
def predecir_curva(nuevos_proyectos):
    nuevos_X_categorical = encoder.transform(nuevos_proyectos[['Pais', 'Sector', 'SubSector', 'TipodePrestamo', 'Categoria Desembolso']])
    nuevos_X_numeric = scaler.transform(nuevos_proyectos[['Monto_Total_Prestamo_Millones']])
    nuevos_X_processed = np.hstack([nuevos_X_categorical, nuevos_X_numeric])

    predicciones = model.predict(nuevos_X_processed)
    predicciones = np.clip(predicciones, 0, 1)
    predicciones = np.maximum.accumulate(predicciones, axis=1)
    
    mask = predicciones >= 0.93
    for i in range(predicciones.shape[0]):
        indices = np.where(mask[i])[0]
        if len(indices) > 1:
            predicciones[i, indices[1]:] = 1.0
    
    return pd.DataFrame(predicciones, columns=Y_columns)

# 📌 Función para graficar la curva con fechas reales
def graficar_curva(predicciones, fecha_inicio, nombre_proyecto):
    fechas = [fecha_inicio + timedelta(days=365 * i) for i in range(len(predicciones.columns))]
    valores = predicciones.iloc[0].values

    plt.figure(figsize=(8, 5))
    plt.plot(fechas, valores, marker='o', linestyle='-', color='b', label='Curva Predicha')

    for i, txt in enumerate(valores):
        plt.text(fechas[i], valores[i], f'{txt:.2f}', ha='right', fontsize=10)

    plt.xlabel("Fecha")
    plt.ylabel("Porcentaje Acumulado")
    plt.title(f"Curva de Desembolso - {nombre_proyecto}")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    return plt

# 📌 Función principal de la app
def app():
    st.title("📊 Predicción de Curvas de Desembolso")

    if model is None:
        return

    st.write("Esta aplicación predice la curva de desembolso acumulado para nuevos proyectos utilizando una red neuronal.")

    nombre_proyecto = st.text_input("🏗️ Nombre del Proyecto:")
    fecha_inicio = st.date_input("📅 Fecha de Inicio del Proyecto:", datetime.today())

    col1, col2 = st.columns(2)
    with col1:
        pais = st.selectbox("🌍 País:", options=unique_values["Pais"])
        sector = st.selectbox("🏭 Sector:", options=unique_values["Sector"])
        subsectores_filtrados = sector_subsector_map.get(sector, [])
        subsector = st.selectbox("🏢 SubSector:", options=subsectores_filtrados)
    
    with col2:
        tipodeprestamo = st.selectbox("💰 Tipo de Préstamo:", options=unique_values["TipodePrestamo"])
        categoria = st.selectbox("📊 Categoría de Desembolso:", options=unique_values["Categoria Desembolso"])
        monto = st.number_input("💵 Monto Total del Préstamo (Millones)", value=40.0)

    if st.button("Predecir Curva"):
        df_nuevos = pd.DataFrame({
            'Pais': [pais],
            'Sector': [sector],
            'SubSector': [subsector],
            'TipodePrestamo': [tipodeprestamo],
            'Categoria Desembolso': [categoria],
            'Monto_Total_Prestamo_Millones': [monto]
        })

        curvas_predichas = predecir_curva(df_nuevos)

        st.subheader("📊 Curvas Predichas")
        st.dataframe(curvas_predichas)

        st.subheader("📈 Gráfico de la Curva de Desembolso")
        plt_obj = graficar_curva(curvas_predichas, fecha_inicio, nombre_proyecto)
        st.pyplot(plt_obj.gcf())

if __name__ == "__main__":
    app()










