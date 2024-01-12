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


def process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos, selected_countries):
    if selected_countries:
        df_operaciones = df_operaciones[df_operaciones['Pais'].isin(selected_countries)]

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
    
    merged_df['Porcentaje'] = ((merged_df['Monto'] / merged_df['AporteFONPLATAVigente']) * 100).round(2)
    merged_df['Monto'] = (merged_df['Monto']/1000).round(0)
    st.write(merged_df)
    return merged_df[merged_df['Ano'] >= 0]

def create_pivot_table(filtered_df, value_column):
    pivot_table = pd.pivot_table(filtered_df, values=value_column, index='IDEtapa', columns='Ano', aggfunc='sum', fill_value=0)
    
    pivot_table['Total'] = pivot_table.sum(axis=1).round(0)
    
    return pivot_table

df_proyectos = load_data(sheet_url_proyectos)
df_operaciones = load_data(sheet_url_operaciones)
df_operaciones_desembolsos = load_data(sheet_url_desembolsos)

unique_countries = df_operaciones['Pais'].unique().tolist()
selected_countries = st.multiselect('Seleccione Países', unique_countries, default=unique_countries)

processed_data = process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos, selected_countries)

pivot_table_monto = create_pivot_table(processed_data, 'Monto')
st.write("Tabla Pivote de Monto de Desembolsos por Proyecto y Año")
st.dataframe(pivot_table_monto)

# Convertir el DataFrame a bytes y agregar botón de descarga
excel_bytes = dataframe_to_excel_bytes(pivot_table_monto)
st.download_button(
    label="Descargar DataFrame en Excel",
    data=excel_bytes,
    file_name="matriz_monto_desembolsos.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

pivot_table_porcentaje = create_pivot_table(processed_data, 'Porcentaje')
st.write("Tabla Pivote de Porcentaje de Desembolsos por Proyecto y Año")
st.dataframe(pivot_table_porcentaje)

# Convertir el DataFrame a bytes y agregar botón de descarga
excel_bytes = dataframe_to_excel_bytes(pivot_table_porcentaje)
st.download_button(
    label="Descargar DataFrame en Excel",
    data=excel_bytes,
    file_name="matriz_porcentaje_desembolsos.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
