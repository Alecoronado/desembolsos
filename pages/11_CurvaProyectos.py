import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import io
from datetime import datetime

# Configuración inicial
st.title("Análisis de Desembolsos por Proyecto")

def load_data():
    """Cargar datos desde Excel local"""
    file_path = "Cartera_Desembolsos.xlsx"
    df = pd.read_excel(file_path)
    return df

def process_data(df):
    """Procesar datos para análisis de curva de desembolsos"""
    
    # Convertir fechas
    df['FechaEfectiva'] = pd.to_datetime(df['FechaEfectiva'], format='%d/%m/%Y', errors='coerce')
    df['FechaVigencia'] = pd.to_datetime(df['FechaVigencia'], format='%d/%m/%Y', errors='coerce')
    
    # Convertir montos a numérico
    df['Monto'] = pd.to_numeric(df['Monto'], errors='coerce')
    df['AporteFONPLATAVigente'] = pd.to_numeric(df['AporteFONPLATAVigente'], errors='coerce')
    
    # Calcular año relativo desde la fecha de vigencia
    df['Ano'] = ((df['FechaEfectiva'] - df['FechaVigencia']).dt.days / 365).fillna(-1)
    
    # Calcular año de fecha efectiva
    df['Ano_FechaEfectiva'] = df['FechaEfectiva'].dt.year
    
    # Calcular porcentaje del monto respecto al aporte total
    df['Porcentaje'] = ((df['Monto'] / df['AporteFONPLATAVigente']) * 100).round(2)
    
    # Filtrar solo registros con Ano >= 0 (desembolsos después de la vigencia)
    filtered_df = df[df['Ano'] >= 0].copy()
    filtered_df['Ano'] = filtered_df['Ano'].astype(int)
    
    # Crear IDOperacion_Alias para el selectbox
    filtered_df['IDOperacion'] = filtered_df['IDOperacion'].astype(str)
    filtered_df['IDOperacion_Alias'] = filtered_df.apply(
        lambda x: f"{x['IDOperacion']} ({x['Alias']})" if pd.notna(x['Alias']) else x['IDOperacion'], 
        axis=1
    )
    
    # Selectbox para filtrar por proyecto
    unique_operaciones_alias = sorted(filtered_df['IDOperacion_Alias'].unique())
    selected_operacion_alias = st.selectbox('Selecciona Proyecto (IDOperacion)', unique_operaciones_alias)
    
    if selected_operacion_alias:
        selected_operacion = selected_operacion_alias.split(' ')[0]
        filtered_result_df = filtered_df[filtered_df['IDOperacion'] == selected_operacion]
    else:
        filtered_result_df = filtered_df
    
    # Agrupar por año relativo
    result_df = filtered_result_df.groupby(['IDOperacion', 'Ano'])[['Monto', 'Porcentaje']].sum().reset_index()
    result_df['Monto'] = (result_df['Monto'] / 1000000).round(2)  # Convertir a millones
    result_df['Monto Acumulado'] = result_df.groupby('IDOperacion')['Monto'].cumsum().round(2)
    result_df['Porcentaje Acumulado'] = result_df.groupby('IDOperacion')['Porcentaje'].cumsum().round(2)
    
    # Agrupar por año de fecha efectiva
    result_df_ano_efectiva = filtered_result_df.groupby(['IDOperacion', 'Ano_FechaEfectiva'])[['Monto', 'Porcentaje']].sum().reset_index()
    result_df_ano_efectiva['Monto'] = (result_df_ano_efectiva['Monto'] / 1000000).round(2)  # Convertir a millones
    result_df_ano_efectiva['Monto Acumulado'] = result_df_ano_efectiva.groupby('IDOperacion')['Monto'].cumsum().round(2)
    result_df_ano_efectiva['Porcentaje Acumulado'] = result_df_ano_efectiva.groupby('IDOperacion')['Porcentaje'].cumsum().round(2)
    
    return result_df, result_df_ano_efectiva

def line_chart(data, x_col, y_col, title, color):
    """Crear gráfico de líneas"""
    chart = alt.Chart(data).mark_line(point=True, color=color).encode(
        x=alt.X(f'{x_col}:O', axis=alt.Axis(title='Año')),
        y=alt.Y(f'{y_col}:Q', axis=alt.Axis(title=y_col)),
        tooltip=[x_col, y_col]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

def run():
    """Función principal"""
    # Cargar datos
    df = load_data()
    
    # Procesar datos
    result_df, result_df_ano_efectiva = process_data(df)
    
    # Mostrar tabla por año relativo
    st.write("### Tabla por Año Relativo (desde Fecha de Vigencia):")
    st.dataframe(result_df)
    
    # Gráficos para Año Relativo
    st.altair_chart(line_chart(result_df, 'Ano', 'Monto', 'Monto por Año en Millones', 'steelblue'))
    st.altair_chart(line_chart(result_df, 'Ano', 'Monto Acumulado', 'Monto Acumulado por Año en Millones', 'goldenrod'))
    st.altair_chart(line_chart(result_df, 'Ano', 'Porcentaje', 'Porcentaje por Año', 'salmon'))
    st.altair_chart(line_chart(result_df, 'Ano', 'Porcentaje Acumulado', 'Porcentaje Acumulado del Monto por Año', 'green'))
    
    # Mostrar tabla por año de fecha efectiva
    st.write("### Tabla por Año de Fecha Efectiva:")
    st.dataframe(result_df_ano_efectiva)
    
    # Gráficos para Año de Fecha Efectiva
    st.altair_chart(line_chart(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Monto', 'Monto por Año de Fecha Efectiva', 'steelblue'))
    st.altair_chart(line_chart(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Monto Acumulado', 'Monto Acumulado por Año de Fecha Efectiva', 'goldenrod'))

if __name__ == "__main__":
    run()

