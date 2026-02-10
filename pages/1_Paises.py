import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 📌 Cargar los datos desde el archivo Excel
def cargar_datos():
    file_path = "Desembolsos_Acum_Max.xlsx"  
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df = df[['Pais', 'Categoria Desembolso', 'Años', 'Porcentaje Acumulado']].dropna()
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró `Desembolsos_Acum_Max.xlsx`. Verifica que esté en la carpeta correcta.")
        return pd.DataFrame()

# 📌 Función para realizar la regresión y graficar resultados
def realizar_regresion(df_filtro, pais_seleccionado, categoria_seleccionada):
    X = df_filtro[['Años']].values
    y = df_filtro['Porcentaje Acumulado'].values

    if len(X) < 2:
        st.warning("⚠ No hay suficientes datos para calcular la regresión.")
        return

    # 📌 Aplicar regresión lineal
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    y_pred_lineal = modelo_lineal.predict(X)
    r2_lineal = r2_score(y, y_pred_lineal)

    # 📌 Aplicar regresión polinómica (grado 2)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)
    y_pred_poly = modelo_poly.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    # 📌 Mostrar los coeficientes R² en columnas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("� R² Regresión Lineal", f"{r2_lineal:.4f}")
    with col2:
        st.metric("📈 R² Regresión Polinómica (grado 2)", f"{r2_poly:.4f}")

    # 📌 Crear puntos suaves para la curva polinómica
    X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_smooth_poly = poly_features.transform(X_smooth)
    y_smooth_poly = modelo_poly.predict(X_smooth_poly)

    # 📌 Crear gráfico en Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', s=100, alpha=0.6, label="Datos Reales", zorder=3)
    ax.plot(X, y_pred_lineal, color='red', linestyle="--", linewidth=2, label=f"Regresión Lineal (R²={r2_lineal:.4f})", zorder=2)
    ax.plot(X_smooth, y_smooth_poly, color='green', linewidth=2, label=f"Regresión Polinómica (R²={r2_poly:.4f})", zorder=2)
    ax.set_xlabel("Años", fontsize=12)
    ax.set_ylabel("Porcentaje Acumulado", fontsize=12)
    ax.set_title(f"Análisis de Regresión para {pais_seleccionado} - {categoria_seleccionada}", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 📌 Mostrar gráfico
    st.pyplot(fig)


# 📌 Función principal de la página
def app():
    st.title("📊 Análisis de Regresión: Porcentaje Acumulado por Años")

    # 📌 Cargar datos
    df = cargar_datos()
    if df.empty:
        return

    # 📌 Selector de país dentro de la app
    paises = sorted(df['Pais'].unique())
    pais_seleccionado = st.selectbox("🌍 Selecciona un país:", paises)

    # 📌 Filtrar las categorías de desembolso según el país seleccionado
    categorias_disponibles = df[df['Pais'] == pais_seleccionado]['Categoria Desembolso'].unique()

    if len(categorias_disponibles) == 0:
        st.warning(f"⚠ No hay categorías de desembolso disponibles para {pais_seleccionado}.")
        return

    categoria_seleccionada = st.selectbox("📊 Selecciona una categoría de desembolso:", sorted(categorias_disponibles))

    # 📌 Filtrar datos por país y categoría de desembolso
    df_filtro = df[(df['Pais'] == pais_seleccionado) & (df['Categoria Desembolso'] == categoria_seleccionada)]

    if df_filtro.empty:
        st.warning(f"⚠ No hay datos disponibles para {pais_seleccionado} - {categoria_seleccionada}.")
        return

    # 📌 Ejecutar la regresión y graficar resultados
    realizar_regresion(df_filtro, pais_seleccionado, categoria_seleccionada)

# 📌 Ejecutar la app si se llama directamente
if __name__ == "__main__":
    app()


