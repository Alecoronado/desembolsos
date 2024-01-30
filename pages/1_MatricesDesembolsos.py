import streamlit as st
import pandas as pd
import numpy as np
import threading
import io

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    excel_bytes = output.getvalue()
    return excel_bytes

# Configuración inicial
LOGGER = st.logger.get_logger(__name__)
_lock = threading.Lock()

# URLs de las hojas de Google Sheets
sheet_url_proyectos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHedheaRLyqnjwtsRvlBFFOnzhfarkFMoJ04chQbKZCBRZXh_2REE3cmsRC69GwsUK0PoOVv95xptX/pub?gid=2084477941&single=true&output=csv"
sheet_url_operaciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHedheaRLyqnjwtsRvlBFFOnzhfarkFMoJ04chQbKZCBRZXh_2REE3cmsRC69GwsUK0PoOVv95xptX/pub?gid=1468153763&single=true&output=csv"
sheet_url_desembolsos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHedheaRLyqnjwtsRvlBFFOnzhfarkFMoJ04chQbKZCBRZXh_2REE3cmsRC69GwsUK0PoOVv95xptX/pub?gid=1657640798&single=true&output=csv"

st.title("Análisis de Desembolsos por Proyecto")

def load_data(url):
    with _lock:
        return pd.read_csv(url)
    
def clean_and_convert_to_float(monto_str):
    if pd.isna(monto_str):
        return np.nan
    try:
        # Asumiendo que 'monto_str' es una cadena, remover puntos de los miles y cambiar comas por puntos para decimales
        cleaned_monto = monto_str.replace('.', '').replace(',', '.')
        return float(cleaned_monto)
    except ValueError:
        # Si hay un error en la conversión, retorna NaN
        return np.nan


def process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos):

    # Aplicar la función de limpieza a la columna 'Monto'
    df_operaciones_desembolsos['Monto'] = df_operaciones_desembolsos['Monto'].apply(clean_and_convert_to_float)
    df_operaciones['AporteFONPLATAVigente'] = df_operaciones['AporteFONPLATAVigente'].apply(clean_and_convert_to_float)

    df_proyectos = df_proyectos[['NoProyecto', 'IDAreaPrioritaria','AreaPrioritaria','IDAreaIntervencion','AreaIntervencion']]
    df_operaciones = df_operaciones[['NoProyecto', 'NoOperacion', 'IDEtapa', 'Alias', 'Pais', 'FechaVigencia', 'Estado', 'AporteFONPLATAVigente']]
    df_operaciones_desembolsos = df_operaciones_desembolsos[['IDDesembolso', 'IDOperacion', 'Monto', 'FechaEfectiva']]

    merged_df = pd.merge(df_operaciones_desembolsos, df_operaciones, left_on='IDOperacion', right_on='IDEtapa', how='left')
    merged_df = pd.merge(merged_df, df_proyectos, on='NoProyecto', how='left')

    merged_df['FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva'], dayfirst=True, errors='coerce')
    merged_df['FechaVigencia'] = pd.to_datetime(merged_df['FechaVigencia'], dayfirst=True, errors='coerce')
    merged_df['Ano'] = ((merged_df['FechaEfectiva'] - merged_df['FechaVigencia']).dt.days / 366).fillna(-1)
    merged_df['Ano'] = merged_df['Ano'].astype(int)
    
    # Convierte las columnas 'Monto' y 'AporteFONPLATAVigente' a numéricas
    merged_df['Monto'] = pd.to_numeric(merged_df['Monto'], errors='coerce')
    merged_df['AporteFONPLATAVigente'] = pd.to_numeric(merged_df['AporteFONPLATAVigente'], errors='coerce')
    
    st.write(merged_df)
    # Convertir el DataFrame a bytes y agregar botón de descarga para ambas tablas
    excel_bytes_monto = dataframe_to_excel_bytes(merged_df)
    st.download_button(
        label="Descargar DataFrame en Excel (Monto)",
        data=excel_bytes_monto,
        file_name="matriz_monto_desembolsos2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



    merged_df['Porcentaje'] = ((merged_df['Monto'] / merged_df['AporteFONPLATAVigente']) * 100).round(2)
    merged_df['Monto'] = (merged_df['Monto']/1000).round(0)
    st.write(merged_df)

    # Convertir el DataFrame a bytes y agregar botón de descarga para ambas tablas
    excel_bytes_monto = dataframe_to_excel_bytes(merged_df)
    st.download_button(
        label="Descargar DataFrame en Excel (Monto)",
        data=excel_bytes_monto,
        file_name="matriz_monto_desembolsos3.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    return merged_df[merged_df['Ano'] >= 0]

def create_pivot_table(filtered_df, value_column):
    pivot_table = pd.pivot_table(filtered_df, values=value_column, index='IDEtapa', columns='Ano', aggfunc='sum', fill_value=0)
    
    pivot_table['Total'] = pivot_table.sum(axis=1).round(0)
    
    return pivot_table

def run():
    df_proyectos = load_data(sheet_url_proyectos)
    df_operaciones = load_data(sheet_url_operaciones)
    df_operaciones_desembolsos = load_data(sheet_url_desembolsos)

    processed_data = process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos)

    # Agregar un filtro multiselect para los países con la opción "Todos"
    paises_disponibles = processed_data['Pais'].unique()  # Obtiene una lista de todos los países únicos
    paises_disponibles = np.insert(paises_disponibles, 0, "Todos")  # Agrega "Todos" al inicio de la lista
    paises_seleccionados = st.multiselect('Seleccionar Países', paises_disponibles, default="Todos")

    # Si se selecciona "Todos", se seleccionan todos los países
    if "Todos" in paises_seleccionados:
        filtered_data = processed_data
    else:
        filtered_data = processed_data[processed_data['Pais'].isin(paises_seleccionados)]

    # Crear y mostrar la tabla pivote de Monto
    pivot_table_monto = create_pivot_table(filtered_data, 'Monto')
    st.write("Tabla Pivote de Monto de Desembolsos por Proyecto y Año")
    st.dataframe(pivot_table_monto)

    # Convertir el DataFrame a bytes y agregar botón de descarga para ambas tablas
    excel_bytes_monto = dataframe_to_excel_bytes(pivot_table_monto)
    st.download_button(
        label="Descargar DataFrame en Excel (Monto)",
        data=excel_bytes_monto,
        file_name="matriz_monto_desembolsos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Crear y mostrar la tabla pivote de Porcentaje
    pivot_table_porcentaje = create_pivot_table(filtered_data, 'Porcentaje')
    st.write("Tabla Pivote de Porcentaje de Desembolsos por Proyecto y Año")
    st.dataframe(pivot_table_porcentaje)

    excel_bytes_porcentaje = dataframe_to_excel_bytes(pivot_table_porcentaje)
    st.download_button(
        label="Descargar DataFrame en Excel (Porcentaje)",
        data=excel_bytes_porcentaje,
        file_name="matriz_porcentaje_desembolsos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    porcentaje_pivot = pivot_table_porcentaje

    # Aplicando la misma lógica para calcular los años hasta ahora y la categorización
    porcentaje_pivot['Años hasta Ahora'] = porcentaje_pivot.iloc[:, 1:10].apply(
            lambda row: row.last_valid_index(), axis=1
        )

        # Creando la función de categorización
    def categorize_project(row):
            if row['Total'] == 100:
                return 'Completado'
            elif row['Total'] >= 50:
                return 'Últimos Desembolsos'
            else:
                return 'Empezando sus Desembolsos'

        # Identificamos las columnas que contienen los porcentajes por año, excluyendo 'Total' y 'Años hasta Ahora'
    year_columns = [col for col in porcentaje_pivot.columns if col not in ['Total', 'Años hasta Ahora']]

        # Encontramos el último año con un valor que no sea cero para cada proyecto
    last_year_with_value = porcentaje_pivot[year_columns].apply(lambda row: row[row > 0].last_valid_index(), axis=1)

        # Agregamos esta información al DataFrame
    porcentaje_pivot['Último Año'] = last_year_with_value

        # Creando la columna de categorización
    porcentaje_pivot['Categoría'] = porcentaje_pivot.apply(categorize_project, axis=1)

        # Restableciendo el índice para convertir 'IDEtapa' de nuevo en una columna
    porcentaje_pivot_reset = porcentaje_pivot.reset_index()

        # Seleccionando las columnas para la tabla final
    final_table_pivot = porcentaje_pivot_reset[['IDEtapa', 'Total', 'Último Año', 'Categoría']]

        # Creando la columna de categorización
    porcentaje_pivot['Categoría'] = porcentaje_pivot.apply(categorize_project, axis=1)

        # Contando el número de proyectos en cada categoría
    category_counts_pivot = porcentaje_pivot['Categoría'].value_counts()

        # Utilizar st.columns para colocar gráficos lado a lado
    col1, col2 = st.columns(2)
    with col1:
            # Mostrando las primeras filas de la tabla final
            final_table_pivot

    with col2:
            # Mostrando las primeras filas de la tabla final
            category_counts_pivot

    # Calcular el promedio del último año
    porcentaje_pivot['Último Año'] = pd.to_numeric(porcentaje_pivot['Último Año'], errors='coerce')
    porcentaje_pivot = porcentaje_pivot[porcentaje_pivot['Total'] == 100]
    filtered_data = porcentaje_pivot.dropna(subset=['Último Año'])
    promedio_ultimo_año = filtered_data['Último Año'].mean().round(0)
    maximo_ultimo_año = filtered_data['Último Año'].max().round(2)


   # Contar la cantidad de proyectos (IDEtapa)
    cantidad_proyectos = porcentaje_pivot.index.nunique()

    # Mostrar el promedio y la cantidad de proyectos en la aplicación Streamlit
    st.subheader('Estadisticas sobre Desembolsos Completados')
    st.write(f'Promedio de Años en Completar: {promedio_ultimo_año}')
    st.write(f'Proyecto que más Años demoro : {maximo_ultimo_año}')
    st.write(f'Cantidad de Proyectos con Desembolsos Completados: {cantidad_proyectos}')


if __name__ == "__main__":
    run()

