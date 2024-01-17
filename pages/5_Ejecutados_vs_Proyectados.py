import streamlit as st
import pandas as pd
import calendar
import altair as alt

# Función para cargar datos desde Google Sheets
def load_data():
    url_operaciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFmOu4IjdEt7gLuAqjJTMvcpelmTr_IsL1WRy238YgRPDGLxsW74iMVUhYM2YegUblAKbLemfMxpW8/pub?gid=0&single=true&output=csv"
    url_proyecciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFmOu4IjdEt7gLuAqjJTMvcpelmTr_IsL1WRy238YgRPDGLxsW74iMVUhYM2YegUblAKbLemfMxpW8/pub?gid=299668301&single=true&output=csv"

    data_operaciones = pd.read_csv(url_operaciones, parse_dates=['FechaEfectiva'], dayfirst=True)
    data_proyecciones = pd.read_csv(url_proyecciones, parse_dates=['Fecha'], dayfirst=True)

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
    merged_data['Ejecutados']= (merged_data['Ejecutados']/1000000).round(3)
    merged_data['Proyectados']= (merged_data['Proyectados']/1000000).round(3)
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

# Function to create and return an Altair line chart with value labels
def create_line_chart_with_labels(data):
    # Melt the DataFrame to long format
    long_df = data.reset_index().melt('index', var_name='Month', value_name='Amount')

    # Define the correct order for months
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]

    # Create a line chart
    line = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X('Month:N', sort=month_order),
        y=alt.Y('Amount:Q', title='Amount'),
        color='index:N',
        tooltip=['Month', 'Amount', 'index']
    ).properties(
        width=450,
        height=500
    )

    # Add text labels for the data points
    text = line.mark_text(
        align='left',
        baseline='middle',
        dx=8,
    ).encode(
        text='Amount:Q'
    )

    return (line + text)

# Función principal de la aplicación Streamlit
def main():
    # Título de la aplicación
    st.title("Análisis de Desembolsos")

    # Cargar datos
    data = load_data()

    # Filtrar por IDOperacion antes de seleccionar el año
    selected_project = st.selectbox("Selecciona proyecto", ["Todos"] + data['IDOperacion'].unique().tolist())

    if selected_project == "Todos":
        filtered_data = data
    else:
        # Filtrar por IDOperacion
        filtered_data = data[data['IDOperacion'] == selected_project]

    # Obtener lista de años únicos basados en los datos filtrados
    unique_years = filtered_data['Year'].unique().tolist()

    # Seleccionar el año mediante un selectbox
    year = st.selectbox("Selecciona el año", unique_years)

    # Obtener datos mensuales para el año seleccionado
    monthly_data = get_monthly_data(filtered_data, year)

    # Mostrar los datos en Streamlit
    st.write(f"Desembolsos Mensuales para {year} - Proyecto seleccionado: {selected_project}")
    st.write(monthly_data)

    # Crear y mostrar el gráfico Altair
    chart = create_line_chart_with_labels(monthly_data)
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()





