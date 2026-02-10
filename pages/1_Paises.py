import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
    ax.set_title(f"Regresión Lineal para {pais_seleccionado} - {categoria_seleccionada}")
    ax.legend()
    
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


