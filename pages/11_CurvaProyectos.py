import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import threading
import io
from datetime import datetime

# Configuración inicial
_lock = threading.Lock()

# URLs de las hojas de Google Sheets
sheet_url_proyectos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHedheaRLyqnjwtsRvlBFFOnzhfarkFMoJ04chQbKZCBRZXh_2REE3cmsRC69GwsUK0PoOVv95xptX/pub?gid=2084477941&single=true&output=csv"
sheet_url_operaciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTadFwCrS_aws658IA94yjGvX_u5oaLnZ8JTVfTZqpaLhI1szZEUbst3rR1rC-zfReRNpMFt93RK_YV/pub?gid=420865954&single=true&output=csv"
sheet_url_desembolsos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTadFwCrS_aws658IA94yjGvX_u5oaLnZ8JTVfTZqpaLhI1szZEUbst3rR1rC-zfReRNpMFt93RK_YV/pub?gid=0&single=true&output=csv"

# Inicializar la aplicación de Streamlit
st.title("Análisis de Desembolsos por Proyecto")

def load_data(url):
    with _lock:
        return pd.read_csv(url)

def convert_to_float(monto_str):
    try:
        monto_str = str(monto_str).replace('.', '').replace(',', '.')
        return float(monto_str)
    except ValueError:
        return np.nan

def process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos):
    df_proyectos = df_proyectos[['NoProyecto', 'IDAreaPrioritaria', 'IDAreaIntervencion']]
    df_operaciones = df_operaciones[['NoProyecto', 'NoOperacion', 'IDEtapa', 'Alias', 'Pais', 'FechaVigencia', 'Estado', 'AporteFONPLATAVigente']]
    df_operaciones_desembolsos = df_operaciones_desembolsos[['IDDesembolso','IDOperacion', 'NoOperacion', 'Monto', 'FechaEfectiva']]
    
    df_operaciones_desembolsos['Monto'] = df_operaciones_desembolsos['Monto'].apply(convert_to_float)
    df_operaciones_desembolsos = df_operaciones_desembolsos.iloc[:, 1:].drop_duplicates()
    
    merged_df = pd.merge(df_operaciones_desembolsos, df_operaciones, left_on='IDOperacion', right_on='IDEtapa', how='left')
    merged_df = pd.merge(merged_df, df_proyectos, on='NoProyecto', how='left')
    
    merged_df['FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva'], dayfirst=True, errors='coerce')
    merged_df['FechaVigencia'] = pd.to_datetime(merged_df['FechaVigencia'], dayfirst=True, errors='coerce')
    merged_df['Ano'] = ((merged_df['FechaEfectiva'] - merged_df['FechaVigencia']).dt.days / 365).fillna(-1)
    merged_df['Ano_FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva']).dt.year
    
    merged_df['Monto'] = pd.to_numeric(merged_df['Monto'], errors='coerce')
    merged_df['AporteFONPLATAVigente'] = pd.to_numeric(merged_df['AporteFONPLATAVigente'], errors='coerce')
    merged_df['Porcentaje'] = ((merged_df['Monto'] / merged_df['AporteFONPLATAVigente']) * 100).round(2)
    filtered_df = merged_df[merged_df['Ano'] >= 0]
    filtered_df['Ano'] = filtered_df['Ano'].astype(int)
    
    etapa_to_alias = df_operaciones.set_index('IDEtapa')['Alias'].to_dict()
    filtered_df['IDEtapa'] = filtered_df['IDEtapa'].astype(str)
    filtered_df['IDEtapa_Alias'] = filtered_df['IDEtapa'].map(lambda x: f"{x} ({etapa_to_alias.get(x, '')})")
    
    unique_etapas_alias = sorted(filtered_df['IDEtapa_Alias'].unique())
    selected_etapa_alias = st.selectbox('Select IDEtapa to filter', unique_etapas_alias)
    
    if selected_etapa_alias:
        selected_etapa = selected_etapa_alias.split(' ')[0]
        filtered_result_df = filtered_df[filtered_df['IDEtapa'] == selected_etapa]
    else:
        filtered_result_df = filtered_df

    result_df = filtered_result_df.groupby(['IDEtapa', 'Ano'])[['Monto', 'Porcentaje']].sum().reset_index()
    result_df['Monto Acumulado'] = result_df.groupby('IDEtapa')['Monto'].cumsum().round(2).reset_index(drop=True)
    result_df['Porcentaje Acumulado'] = result_df.groupby('IDEtapa')['Porcentaje'].cumsum().round(2).reset_index(drop=True)

    result_df_ano_efectiva = filtered_result_df.groupby(['IDEtapa', 'Ano_FechaEfectiva'])[['Monto', 'Porcentaje']].sum().reset_index()
    result_df_ano_efectiva['Monto Acumulado'] = result_df_ano_efectiva.groupby('IDEtapa')['Monto'].cumsum().round(2).reset_index(drop=True)
    result_df_ano_efectiva['Porcentaje Acumulado'] = result_df_ano_efectiva.groupby('IDEtapa')['Porcentaje'].cumsum().round(2).reset_index(drop=True)

    return result_df, result_df_ano_efectiva

def line_chart(data, x_col, y_col, title, color):
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
    df_proyectos = load_data(sheet_url_proyectos)
    df_operaciones = load_data(sheet_url_operaciones)
    df_operaciones_desembolsos = load_data(sheet_url_desembolsos)
    
    result_df, result_df_ano_efectiva = process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos)
    
    st.write("Tabla por Año:", result_df)
    
    # Gráficos para Año
    st.altair_chart(line_chart(result_df, 'Ano', 'Monto', 'Monto por Año en Millones', 'steelblue'))
    st.altair_chart(line_chart(result_df, 'Ano', 'Monto Acumulado', 'Monto Acumulado por Año en Millones', 'goldenrod'))
    st.altair_chart(line_chart(result_df, 'Ano', 'Porcentaje', 'Porcentaje por Año', 'salmon'))
    st.altair_chart(line_chart(result_df, 'Ano', 'Porcentaje Acumulado', 'Porcentaje Acumulado del Monto por Año', 'green'))

    st.write("Tabla por Año de Fecha Efectiva:", result_df_ano_efectiva)
    # Gráficos para Año de Fecha Efectiva
    st.altair_chart(line_chart(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Monto', 'Monto por Año de Fecha Efectiva', 'steelblue'))
    st.altair_chart(line_chart(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Monto Acumulado', 'Monto Acumulado por Año de Fecha Efectiva', 'goldenrod'))

if __name__ == "__main__":
    run()
