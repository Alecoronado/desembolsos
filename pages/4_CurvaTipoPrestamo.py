import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 📌 Cargar los datos desde el archivo Excel
def cargar_datos():
    file_path = "Desembolsos_Acum_Max.xlsx"  # Asegúrate de que el archivo esté en la misma carpeta
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df = df[['TipodePrestamo', 'Categoria Desembolso', 'Años', 'Porcentaje Acumulado']].dropna()
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró `Desembolsos_Acum_Max.xlsx`. Verifica que esté en la carpeta correcta.")
        return pd.DataFrame()

# 📌 Función para realizar la regresión y graficar resultados
def realizar_regresion(df_filtro, prestamo_seleccionado, categoria_seleccionada):
    X = df_filtro[['Años']].values
    y = df_filtro['Porcentaje Acumulado'].values

    if len(X) < 2:
        st.warning("⚠ No hay suficientes datos para calcular la regresión.")
        return

    # 📌 Aplicar regresión lineal
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    # 📌 Mostrar el coeficiente R²
    st.write(f"### 📌 Coeficiente de determinación R²: `{r2:.2f}`")

    # 📌 Crear gráfico en Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color='blue', label="Datos Reales")
    ax.plot(X, y_pred, color='red', linestyle="--", label="Regresión Lineal")
    ax.set_xlabel("Años")
    ax.set_ylabel("Porcentaje Acumulado")
    ax.set_title(f"Regresión Lineal para {prestamo_seleccionado} - {categoria_seleccionada}")
    ax.legend()
    
    # 📌 Mostrar gráfico
    st.pyplot(fig)

# 📌 Función principal de la página
def app():
    st.title("📊 Análisis de Regresión: Porcentaje Acumulado por Años - Tipo de Préstamo")

    # 📌 Cargar datos
    df = cargar_datos()
    if df.empty:
        return

    # 📌 Selector de tipo de préstamo dentro de la app
    tipos_prestamo = sorted(df['TipodePrestamo'].dropna().unique())
    prestamo_seleccionado = st.selectbox("💰 Selecciona un Tipo de Préstamo:", tipos_prestamo)

    # 📌 Filtrar categorías de desembolso según el tipo de préstamo seleccionado
    df_prestamo = df[df['TipodePrestamo'] == prestamo_seleccionado]

    if df_prestamo.empty:
        st.warning(f"⚠ No hay datos disponibles para el Tipo de Préstamo {prestamo_seleccionado}.")
        return

    categorias_disponibles = sorted(df_prestamo['Categoria Desembolso'].dropna().unique())

    if not categorias_disponibles:
        st.warning(f"⚠ No hay categorías de desembolso disponibles para {prestamo_seleccionado}.")
        return

    categoria_seleccionada = st.selectbox("📊 Selecciona una categoría de desembolso:", categorias_disponibles)

    # 📌 Filtrar datos por tipo de préstamo y categoría de desembolso
    df_filtro = df_prestamo[df_prestamo['Categoria Desembolso'] == categoria_seleccionada]

    if df_filtro.empty:
        st.warning(f"⚠ No hay datos disponibles para {prestamo_seleccionado} - {categoria_seleccionada}.")
        return

    # 📌 Ejecutar la regresión y graficar resultados
    realizar_regresion(df_filtro, prestamo_seleccionado, categoria_seleccionada)

# 📌 Ejecutar la app si se llama directamente
if __name__ == "__main__":
    app()

