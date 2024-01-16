import streamlit as st
import pandas as pd
import calendar

# Función para cargar datos desde Google Sheets
def load_data():
    url_operaciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFmOu4IjdEt7gLuAqjJTMvcpelmTr_IsL1WRy238YgRPDGLxsW74iMVUhYM2YegUblAKbLemfMxpW8/pub?gid=0&single=true&output=csv"
    url_proyecciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFmOu4IjdEt7gLuAqjJTMvcpelmTr_IsL1WRy238YgRPDGLxsW74iMVUhYM2YegUblAKbLemfMxpW8/pub?gid=299668301&single=true&output=csv"

    data_operaciones = pd.read_csv(url_operaciones, parse_dates=['FechaEfectiva'])
    data_proyecciones = pd.read_csv(url_proyecciones, parse_dates=['Fecha'])

    data_operaciones['Monto'] = pd.to_numeric(data_operaciones['Monto'], errors='coerce').fillna(0)
    data_proyecciones['Monto'] = pd.to_numeric(data_proyecciones['Monto'], errors='coerce').fillna(0)
    data_operaciones['Ejecutados'] = data_operaciones['Monto']
    data_proyecciones['Proyectados'] = data_proyecciones['Monto']

    data_operaciones['Year'] = data_operaciones['FechaEfectiva'].dt.year
    data_operaciones['Month'] = data_operaciones['FechaEfectiva'].dt.month
    data_proyecciones['Year'] = data_proyecciones['Fecha'].dt.year
    data_proyecciones['Month'] = data_proyecciones['Fecha'].dt.month

    grouped_operaciones = data_operaciones.groupby(['IDOperacion', 'Year', 'Month']).agg({'Monto': 'sum'}).rename(columns={'Monto': 'Ejecutados'}).reset_index()
    grouped_proyecciones = data_proyecciones.groupby(['IDOperacion', 'Year', 'Month']).agg({'Monto': 'sum'}).rename(columns={'Monto': 'Proyectados'}).reset_index()

    merged_data = pd.merge(grouped_operaciones, grouped_proyecciones, on=['IDOperacion', 'Year', 'Month'], how='outer').fillna(0)
    st.write(merged_data)
    return merged_data


# Función para obtener los datos transpuestos por mes para un año específico
def get_monthly_data(data, year):
    data_year = data[data['Year'] == year]

    # Agrupar los datos por mes y sumar los montos
    grouped_data = data_year.groupby('Month').agg({'Proyectados': 'sum', 'Ejecutados': 'sum'}).reset_index()

    # Reemplazar el número del mes con el nombre del mes
    grouped_data['Month'] = grouped_data['Month'].apply(lambda x: calendar.month_name[x])

    # Transponer el DataFrame para que los meses sean las columnas y las filas sean 'Proyectados' y 'Ejecutados'
    transposed_data = grouped_data.set_index('Month').T

    return transposed_data

# Función principal de la aplicación Streamlit
def main():
    st.title("Análisis de Desembolsos")

    data = load_data()

    year = st.selectbox("Selecciona el año", options=[2024, 2025, 2026])

    monthly_data = get_monthly_data(data, year)

    # Mostrar los datos en Streamlit
    st.write(f"Desembolsos Mensuales para {year}:")
    st.write(monthly_data)

if __name__ == "__main__":
    main()

