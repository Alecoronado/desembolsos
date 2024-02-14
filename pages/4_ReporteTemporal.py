import streamlit as st
import pandas as pd
import numpy as np
import threading
import io
import matplotlib.pyplot as plt
import altair as alt

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    excel_bytes = output.getvalue()
    return excel_bytes

LOGGER = st.logger.get_logger(__name__)
_lock = threading.Lock()

sheet_url_proyectos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHedheaRLyqnjwtsRvlBFFOnzhfarkFMoJ04chQbKZCBRZXh_2REE3cmsRC69GwsUK0PoOVv95xptX/pub?gid=2084477941&single=true&output=csv"
sheet_url_operaciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHedheaRLyqnjwtsRvlBFFOnzhfarkFMoJ04chQbKZCBRZXh_2REE3cmsRC69GwsUK0PoOVv95xptX/pub?gid=1468153763&single=true&output=csv"
sheet_url_desembolsos = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTadFwCrS_aws658IA94yjGvX_u5oaLnZ8JTVfTZqpaLhI1szZEUbst3rR1rC-zfReRNpMFt93RK_YV/pub?gid=0&single=true&output=csv"

st.title(" Reporte de Desembolsos")

def load_data(url):
    with _lock:
        return pd.read_csv(url)
    
def clean_and_convert_to_float(monto_str):
    if pd.isna(monto_str):
        return np.nan
    try:
        cleaned_monto = monto_str.replace('.', '').replace(',', '.')
        return float(cleaned_monto)
    except ValueError:
        return np.nan

def process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos):
    df_operaciones_desembolsos['Monto'] = df_operaciones_desembolsos['Monto'].apply(clean_and_convert_to_float)
    df_operaciones['AporteFONPLATAVigente'] = df_operaciones['AporteFONPLATAVigente'].apply(clean_and_convert_to_float)

    df_proyectos = df_proyectos[['NoProyecto', 'IDAreaPrioritaria','AreaPrioritaria','IDAreaIntervencion','AreaIntervencion','Responsable']]
    df_operaciones = df_operaciones[['NoProyecto', 'NoOperacion', 'IDEtapa', 'Alias', 'Pais', 'FechaVigencia', 'Estado', 'AporteFONPLATAVigente']]
    df_operaciones_desembolsos = df_operaciones_desembolsos[['IDDesembolso', 'IDOperacion', 'Monto', 'FechaEfectiva']]

    merged_df = pd.merge(df_operaciones_desembolsos, df_operaciones, left_on='IDOperacion', right_on='IDEtapa', how='left')
    merged_df = pd.merge(merged_df, df_proyectos, on='NoProyecto', how='left')

    merged_df['FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva'], dayfirst=True, errors='coerce')
    merged_df['FechaVigencia'] = pd.to_datetime(merged_df['FechaVigencia'], dayfirst=True, errors='coerce')
    merged_df['Ano'] = ((merged_df['FechaEfectiva'] - merged_df['FechaVigencia']).dt.days / 366).fillna(-1)
    merged_df['Ano'] = merged_df['Ano'].astype(int)
    
    merged_df['Monto'] = pd.to_numeric(merged_df['Monto'], errors='coerce')
    merged_df['AporteFONPLATAVigente'] = pd.to_numeric(merged_df['AporteFONPLATAVigente'], errors='coerce')
    
    merged_df['Porcentaje'] = ((merged_df['Monto'] / merged_df['AporteFONPLATAVigente']) * 100).round(2)
    merged_df['Monto'] = (merged_df['Monto']/1000000).round(3)
    

    merged_df['FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva'])
    merged_df['Año'] = merged_df['FechaEfectiva'].dt.year
    merged_df['Mes'] = merged_df['FechaEfectiva'].dt.month

    # Lista de nombres de meses con opción 'Todos los Meses'
    nombres_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    nombres_meses_con_todos = ['Todos los Meses'] + nombres_meses

    # Lista de países con opción 'Todos'
    paises_unicos = list(np.sort(merged_df['Pais'].unique()))
    paises_con_todos = ['Todos'] + paises_unicos

    # Multiselect para elegir países
    selected_countries = st.multiselect(
        'Selecciona países:', 
        options=paises_unicos,
        default=paises_unicos
    )

    # Comprobar si el usuario seleccionó la opción 'Todos'
    if 'Todos' in selected_countries:
        selected_countries = paises_unicos

    # Lógica de filtrado en función de los países seleccionados
    df_filtrado = merged_df[merged_df['Pais'].isin(selected_countries)]

    # Obtener años únicos del DataFrame
    años_unicos = list(np.sort(df_filtrado['Año'].unique()))

    # Slider para elegir año
    año_seleccionado = st.slider(
        'Selecciona un año', 
        min_value=min(años_unicos),
        max_value=max(años_unicos),
        value=min(años_unicos)  # Valor inicial del slider
    )

    # Filtro por año
    df_filtrado = df_filtrado[df_filtrado['Año'] == año_seleccionado]
    # Select box para elegir mes
    mes_seleccionado = st.selectbox(
        'Selecciona un mes', 
        options=nombres_meses_con_todos,
        index=0
    )

    # Filtro por mes
    if mes_seleccionado != 'Todos los Meses':
        mes_numero = nombres_meses.index(mes_seleccionado) + 1
        df_filtrado = df_filtrado[df_filtrado['Mes'] == mes_numero]

    # Select box para elegir sector
    Sector_seleccionado = st.selectbox(
        'Selecciona un Sector', 
        options=['Todos'] + list(np.sort(df_filtrado['IDAreaPrioritaria'].unique())),
        index=0
    )

    # Filtro por sector
    df_filtrado = df_filtrado if Sector_seleccionado == 'Todos' else df_filtrado[df_filtrado['IDAreaPrioritaria'] == Sector_seleccionado]

    st.write(df_filtrado)
    resumen_df = df_filtrado.groupby(['IDAreaPrioritaria', 'AreaPrioritaria']).agg(
    Proyectos=('IDEtapa', 'nunique'),
    Suma_Monto=('Monto', 'sum')
    ).reset_index()


    st.write("Tabla de Resumen de Sectores")

    total_proyectos = resumen_df['Proyectos'].sum()
    total_suma_monto = resumen_df['Suma_Monto'].sum()
    total_resumen_df = pd.DataFrame({'IDAreaPrioritaria': ['Total'], 'Proyectos': [total_proyectos], 'Suma_Monto': [total_suma_monto]})
    resumen_df = pd.concat([resumen_df, total_resumen_df], ignore_index=True)
    st.write(resumen_df)

    excel_bytes_porcentaje = dataframe_to_excel_bytes(resumen_df)
    st.download_button(
        label="Descargar Tabla de Datos Filtrados por Sector",
        data=excel_bytes_porcentaje,
        file_name="AnálisisdeDesembolsosSector.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Creando el gráfico circular con mejoras
    df_pie = resumen_df[resumen_df['IDAreaPrioritaria'] != 'Total']
    fig, ax = plt.subplots()

    # Crear el gráfico de pastel y mostrar porcentajes
    pie, _, autopcts = ax.pie(
        df_pie['Suma_Monto'], 
        labels=df_pie['IDAreaPrioritaria'], 
        autopct='%1.1f%%',  # Formato de porcentaje con un decimal
        startangle=140,
        colors=plt.cm.Paired.colors
    )

    # Añadir un círculo en el centro para un diseño de donut
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    # Establecer un título para el gráfico
    ax.set_title('Distribución de Montos por Sector en Porcentajes', fontsize=14)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
    
    resumen_intervencion_total_df = df_filtrado.groupby(['IDAreaIntervencion', 'AreaIntervencion']).agg(
    Proyectos_Unicos=('IDEtapa', 'nunique'),
    Suma_Monto=('Monto', 'sum')
    ).reset_index()



    st.write("Tabla de Resumen de SubSectores")

    total_proyectos_unicos = resumen_intervencion_total_df['Proyectos_Unicos'].sum()
    total_suma_monto_intervencion = resumen_intervencion_total_df['Suma_Monto'].sum()
    total_resumen_intervencion_df = pd.DataFrame({'IDAreaIntervencion': ['Total'], 'Proyectos_Unicos': [total_proyectos_unicos], 'Suma_Monto': [total_suma_monto_intervencion]})
    resumen_intervencion_total_df = pd.concat([resumen_intervencion_total_df, total_resumen_intervencion_df], ignore_index=True)
    st.write(resumen_intervencion_total_df)

    excel_bytes_porcentaje = dataframe_to_excel_bytes(resumen_intervencion_total_df)
    st.download_button(
        label="Descargar Tabla de Datos Filtrados",
        data=excel_bytes_porcentaje,
        file_name="AnálisisdeDesembolsosSubSector.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Asegúrate de que df_pies esté definido
    df_pies = resumen_intervencion_total_df[resumen_intervencion_total_df['IDAreaIntervencion'] != 'Total']

    # Ordenar los datos por 'Suma_Monto' de menor a mayor
    df_pies = df_pies.sort_values(by='Suma_Monto')

    # Crear el gráfico de barras
    bar_chart = alt.Chart(df_pies).mark_bar().encode(
        x=alt.X('IDAreaIntervencion', sort=None),  # Usar el orden del DataFrame
        y='Suma_Monto',
        color='IDAreaIntervencion',  # Colores distintos para cada IDAreaIntervencion
        tooltip=['IDAreaIntervencion', 'Suma_Monto']
    )

    # Crear el gráfico de texto para las etiquetas
    text_chart = bar_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5  # Ajustar la posición del texto por encima de las barras
    ).encode(
        text='Suma_Monto'
    )

    # Combinar ambos gráficos
    final_chart = (bar_chart + text_chart).properties(
        title='Distribución de Montos por SubSector'
    )

    # Mostrar el gráfico en Streamlit
    st.altair_chart(final_chart, use_container_width=True)
    return df_filtrado

def run():
    # Cargar y procesar los datos
    df_proyectos = load_data(sheet_url_proyectos)
    df_operaciones = load_data(sheet_url_operaciones)
    df_operaciones_desembolsos = load_data(sheet_url_desembolsos)

    processed_data = process_data(df_proyectos, df_operaciones, df_operaciones_desembolsos)

if __name__ == "__main__":
    run()