import streamlit as st
import pandas as pd
import calendar
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import io

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    excel_bytes = output.getvalue()
    return excel_bytes

# Función para cargar datos desde Excel local
def load_data():
    # Cargar el archivo Excel
    file_path = "Cartera_Desembolsos.xlsx"
    df = pd.read_excel(file_path)
    
    # Convertir la columna de fecha
    df['FechaEfectiva'] = pd.to_datetime(df['FechaEfectiva'], format='%d/%m/%Y', errors='coerce')
    df['Monto'] = pd.to_numeric(df['Monto'], errors='coerce')
    
    # Crear columnas de Year y Month
    df['Year'] = df['FechaEfectiva'].dt.year
    df['Month'] = df['FechaEfectiva'].dt.month
    
    # Separar los datos por Estado_desembolso
    data_ejecutados = df[df['Estado_desembolso'] == 'Ejecutado'].copy()
    data_proyectados = df[df['Estado_desembolso'] == 'Proyectado'].copy()
    data_proyecciones_iniciales = df[df['Estado_desembolso'] == 'Proyeccion Inicial'].copy()
    
    # Renombrar columnas de Monto según el tipo
    data_ejecutados['Ejecutados'] = data_ejecutados['Monto']
    data_proyectados['Proyectados'] = data_proyectados['Monto']
    data_proyecciones_iniciales['ProyeccionesIniciales'] = data_proyecciones_iniciales['Monto']
    
    # Agrupar datos por las columnas necesarias (incluyendo Riesgo)
    grouped_ejecutados = data_ejecutados.groupby(['Pais', 'IDOperacion', 'Responsable', 'Riesgo', 'Year', 'Month', 'Sector', 'Alias']).agg({'Ejecutados': 'sum'}).reset_index()
    grouped_proyectados = data_proyectados.groupby(['Pais', 'IDOperacion', 'Responsable', 'Riesgo', 'Year', 'Month', 'Sector', 'Alias']).agg({'Proyectados': 'sum'}).reset_index()
    grouped_proyecciones_iniciales = data_proyecciones_iniciales.groupby(['Pais', 'IDOperacion', 'Responsable', 'Riesgo', 'Year', 'Month', 'Sector', 'Alias']).agg({'ProyeccionesIniciales': 'sum'}).reset_index()
    
    # Combinar los tres conjuntos de datos
    merged_data = pd.merge(grouped_ejecutados, grouped_proyectados, 
                          on=['Pais', 'IDOperacion', 'Responsable', 'Riesgo', 'Year', 'Month', 'Sector', 'Alias'], 
                          how='outer')
    merged_data = pd.merge(merged_data, grouped_proyecciones_iniciales, 
                          on=['Pais', 'IDOperacion', 'Responsable', 'Riesgo', 'Year', 'Month', 'Sector', 'Alias'], 
                          how='outer').fillna(0)
    
    # Conversiones finales y ajustes de escala (convertir a millones)
    merged_data['Ejecutados'] = (merged_data['Ejecutados'] / 1000000).round(2)
    merged_data['Proyectados'] = (merged_data['Proyectados'] / 1000000).round(2)
    merged_data['ProyeccionesIniciales'] = (merged_data['ProyeccionesIniciales'] / 1000000).round(2)
    
    return merged_data


def get_monthly_data(data, year):
    data_year = data[data['Year'] == year]

    # Agrupar los datos por mes y sumar los montos
    grouped_data = data_year.groupby('Month').agg({'Ejecutados': 'sum',
                                                   'Proyectados': 'sum',
                                                   'ProyeccionesIniciales': 'sum'}).reset_index()

    # Reemplazar el número del mes con el nombre del mes en español
    spanish_months = [calendar.month_name[i].capitalize() for i in range(1, 13)]
    grouped_data['Month'] = grouped_data['Month'].apply(lambda x: spanish_months[int(x) - 1])

    # Transponer el DataFrame para que los meses sean columnas y 'Proyectados' y 'Ejecutados' sean las filas
    transposed_data = grouped_data.set_index('Month').T

    # Calcular los totales para cada fila
    transposed_data['Totales'] = transposed_data.sum(axis=1)

    return transposed_data

def create_line_chart_with_labels(data):
    # Filtrar solo las primeras 3 filas para el gráfico
    if data.shape[0] > 3:
        data = data.iloc[:3, :]

    # Eliminar la columna 'Totales' si está presente
    if 'Totales' in data.columns:
        data = data.drop(columns=['Totales'])

    # Convertir el DataFrame a formato largo
    long_df = data.reset_index().melt('index', var_name='Month', value_name='Amount')

    # Definir el orden correcto de los meses en español
    month_order_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                      "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    # Definir los colores para cada línea
    color_scale = alt.Scale(domain=['Ejecutados', 'Proyectados', 'ProyeccionesIniciales'],
                            range=['red', 'blue', 'skyblue'])

    # Crear el gráfico de líneas
    line = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X('Month:N', sort=month_order_es),  
        y=alt.Y('Amount:Q', title='Amount'),
        color=alt.Color('index:N', scale=color_scale),
        tooltip=['Month', 'Amount', 'index']
    ).properties(
        width=600,
        height=400
    )

    # Agregar etiquetas de texto para los puntos de datos
    text = line.mark_text(
        align='left',
        baseline='middle',
        dx=7,
        dy=-15  # Ajusta para evitar solapamiento con los puntos
    ).encode(
        text=alt.Text('Amount:Q', format=".2f")  # Formatear a dos decimales si es necesario
    )

    # Combinar el gráfico de líneas con las etiquetas de texto
    chart = line + text

    return chart

def calculate_cumulative_sum(data):
    # Asegurarse de trabajar solo con las tres filas originales en caso de que haya más filas
    if data.shape[0] > 3:
        data = data.iloc[:3, :]

    # Calcula la suma acumulada a lo largo de los meses para cada una de las filas
    # Asegúrate de no incluir la columna de Totales en el cálculo
    data_cumsum = data.drop(columns=['Totales'], errors='ignore').cumsum(axis=1)

    # Agregar la fila de totales al final si se necesita
    data_cumsum['Totales'] = data_cumsum.iloc[:, -1]

    return data_cumsum


def create_cumulative_line_chart(data):
    # Convertir los valores a millones y redondear a enteros
    data_in_millions = data.apply(lambda x: (x / 1).round())

    # Filtrar solo las primeras 3 filas para el gráfico
    if data_in_millions.shape[0] > 3:
        data_in_millions = data_in_millions.iloc[:3, :]

    # Eliminar la columna 'Totales' si está presente
    if 'Totales' in data_in_millions.columns:
        data_in_millions = data_in_millions.drop(columns=['Totales'])
    
    # Convertir el DataFrame a formato largo
    long_df = data_in_millions.reset_index().melt('index', var_name='Month', value_name='Cumulative Amount')

    # Definir el orden correcto de los meses en español
    month_order_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                      "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    # Definir los colores para cada línea
    color_scale = alt.Scale(domain=['Ejecutados', 'Proyectados', 'ProyeccionesIniciales'],
                            range=['red', 'blue', 'skyblue'])

    # Crear el gráfico de líneas
    line = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X('Month:N', sort=month_order_es),  
        y=alt.Y('Cumulative Amount:Q', title='Cumulative Amount (millions)'),
        color=alt.Color('index:N', scale=color_scale),
        tooltip=['Month', 'Cumulative Amount', 'index']
    ).properties(
        width=600,
        height=400
    )

    # Agregar etiquetas de texto para los puntos de datos
    text = line.mark_text(
        align='left',
        baseline='middle',
        dx=7,
        dy=-15  # Ajusta para evitar solapamiento con los puntos
    ).encode(
        text=alt.Text('Cumulative Amount:Q')  # Se omiten los decimales en el gráfico
    )

    # Combinar el gráfico de líneas con las etiquetas de texto
    chart = line + text

    return chart



def create_comparison_bar_chart(filtered_data, year):
    # Filtrar los datos para el año seleccionado
    data_year = filtered_data[filtered_data['Year'] == year]

    # Agrupar los datos por 'Pais' y calcular la suma de 'Ejecutados' y 'Proyectados', redondeando a un decimal
    grouped_data = data_year.groupby('Pais', as_index=False).agg({
        'Ejecutados': lambda x: round(x.sum(), 2),
        'ProyeccionesIniciales': lambda x: round(x.sum(), 2)
    })

    # Configurar las posiciones y ancho de las barras
    bar_width = 0.4
    index = np.arange(len(grouped_data['Pais']))

    # Iniciar la creación del gráfico
    fig, ax = plt.subplots()

    # Crear las barras para 'Ejecutados'
    bars1 = ax.bar(index - bar_width/2, grouped_data['Ejecutados'], bar_width, label='Ejecutados', color='r')

    # Crear las barras para 'Proyectados'
    bars2 = ax.bar(index + bar_width/2, grouped_data['ProyeccionesIniciales'], bar_width, label='ProyeccionesIniciales', color='skyblue')

    # Añadir las etiquetas de los datos en las barras
    ax.bar_label(bars1, padding=3, fontsize=8, fmt='%.2f')  # Reducir el tamaño de la fuente aquí
    ax.bar_label(bars2, padding=3, fontsize=8, fmt='%.2f')  # Reducir el tamaño de la fuente aquí

    # Ajustar las etiquetas y títulos
    ax.set_xlabel('País')
    ax.set_ylabel('Monto (en millones)')
    ax.set_title('Ejecutados y ProyeccionesIniciales por País')
    ax.set_xticks(index)
    ax.set_xticklabels(grouped_data['Pais'], rotation=45, fontsize=8)  # Reducir el tamaño de la fuente aquí
    ax.set_yticklabels(ax.get_yticks(), fontsize=8)  # Reducir el tamaño de la fuente aquí
    ax.legend()

    # Ajuste final para asegurar que la disposición de las etiquetas sea legible
    plt.subplots_adjust(bottom=0.15)  # Ajustar si es necesario
    fig.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


def create_responsible_comparison_chart(filtered_data, year):
    # Filtrar los datos para el año seleccionado y que tengan valores
    data_year = filtered_data[(filtered_data['Year'] == year) & ((filtered_data['Ejecutados'] > 0) | (filtered_data['ProyeccionesIniciales'] > 0))]

    # Agrupar los datos por 'Responsable'
    grouped_data = data_year.groupby('Responsable', as_index=False).agg({
        'Ejecutados': lambda x: round(x.sum(), 1),
        'ProyeccionesIniciales': lambda x: round(x.sum(), 1)
    })

    # Configurar las posiciones y ancho de las barras
    bar_width = 0.4  # Ancho de las barras
    bar_space = 0.2  # Espacio entre grupos de barras
    index = np.arange(len(grouped_data['Responsable']))

    # Iniciar la creación del gráfico
    fig, ax = plt.subplots()

    # Crear las barras para 'Ejecutados'
    bars1 = ax.bar(index - bar_width/2, grouped_data['Ejecutados'], bar_width, label='Ejecutados', color='r')

    # Crear las barras para 'Proyectados'
    bars2 = ax.bar(index + bar_width/2, grouped_data['ProyeccionesIniciales'], bar_width, label='ProyeccionesIniciales', color='skyblue')

    # Añadir las etiquetas en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:  # Solo añadir etiqueta si el valor es mayor a cero
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.9, round(yval, 1), va='bottom', ha='center', fontsize=5)
    # Añadir las etiquetas y títulos
    ax.set_xlabel('Responsable')
    ax.set_ylabel('Monto')
    ax.set_title('Ejecutados vs ProyeccionesIniciales por Responsable')
    
    # Ajustar las etiquetas del eje x para alinear con las barras
    ax.set_xticks(index)
    # Aumentar la rotación a 90 grados y ajustar la alineación y el tamaño de la fuente
    ax.set_xticklabels(grouped_data['Responsable'], rotation=90, ha='right', fontsize=6, rotation_mode='anchor')

    ax.legend()

    # Ajuste final para asegurar que la disposición de las etiquetas sea legible
    plt.subplots_adjust(bottom=0.5)  # Ajustar el espacio en la parte inferior para dar más espacio a las etiquetas
    fig.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


# Función principal de la aplicación Streamlit
def main():
    # Título de la aplicación
    st.title("Seguimiento de Desembolsos Proyectados")

    # Cargar datos
    data = load_data()

    # Obtener lista de años únicos basados en los datos filtrados
    unique_years = data['Year'].unique().tolist()

    # Filtrar por Pais con selección múltiple
    selected_countries = st.multiselect("Selecciona país(es)", ["Todos"] + data['Pais'].unique().tolist())

    if "Todos" in selected_countries:
        filtered_data = data
    else:
        # Filtrar por países seleccionados
        filtered_data = data[data['Pais'].isin(selected_countries)]

    # Filtrar por Riesgo
    selected_risk = st.selectbox("Selecciona Riesgo", ["Ambos", "RS", "RNS"])
    
    if selected_risk != "Ambos":
        # Filtrar por el riesgo seleccionado
        filtered_data = filtered_data[filtered_data['Riesgo'] == selected_risk]

    # Convertir los valores de año a enteros y obtener la lista ordenada
    unique_years_filtered = sorted(filtered_data['Year'].astype(int).unique())

    # Asegurarse de que haya años disponibles
    if unique_years_filtered:
        # Intentar establecer 2025 como el año predeterminado si está disponible
        default_year = 2025 if 2025 in unique_years_filtered else unique_years_filtered[0]

        # Seleccionar el año mediante un slider
        year = st.slider("Selecciona el año", 
                         min_value=min(unique_years_filtered), 
                         max_value=max(unique_years_filtered), 
                         value=default_year)
    else:
        st.error("No hay datos disponibles para mostrar basados en la selección de país.")
        return 

    # Añadir el filtro de Sector aquí
    unique_sectors = sorted(filtered_data['Sector'].unique().tolist())
    selected_sector = st.selectbox("Selecciona sector", ["Todos"] + unique_sectors)

    if selected_sector != "Todos":
        # Filtrar los datos por el sector seleccionado
        filtered_data = filtered_data[filtered_data['Sector'] == selected_sector]

    # Añadir el filtro de Responsable aquí
    unique_responsibles = sorted(filtered_data['Responsable'].drop_duplicates().tolist())
    selected_responsible = st.selectbox("Selecciona responsable", ["Todos"] + unique_responsibles)

    if selected_responsible != "Todos":
        # Filtrar los datos por el responsable seleccionado
        filtered_data = filtered_data[filtered_data['Responsable'] == selected_responsible]

    # Añadir alias a IDOperacion después de cualquier filtrado previo
    operacion_to_alias = filtered_data.set_index('IDOperacion')['Alias'].to_dict()
    filtered_data['IDOperacion'] = filtered_data['IDOperacion'].astype(str)
    filtered_data['IDOperacion_Alias'] = filtered_data['IDOperacion'].map(lambda x: f"{x} ({operacion_to_alias.get(x, '')})")

    # Ahora, utilizar IDOperacion_Alias para la selección del proyecto en lugar de IDOperacion
    unique_operacion_alias = sorted(filtered_data['IDOperacion_Alias'].unique())
    selected_project_alias = st.selectbox("Selecciona proyecto", ["Todos"] + unique_operacion_alias)

    if selected_project_alias != "Todos":
        # Extraer el IDOperacion del alias seleccionado para filtrar
        selected_project = selected_project_alias.split(" (")[0]  # Asumiendo que el IDOperacion está antes del primer paréntesis
        filtered_data = filtered_data[filtered_data['IDOperacion'] == selected_project]
    else:
        # Si se selecciona "Todos", no es necesario filtrar más
        pass

    # Obtener datos mensuales para el año seleccionado
    monthly_data = get_monthly_data(filtered_data, year)

    # Mostrar los datos en Streamlit
    st.write(f"Desembolsos Mensuales para {year} - País(es) seleccionado(s): {', '.join(selected_countries)}")
    st.write(monthly_data)

    # Convertir el DataFrame a bytes y agregar botón de descarga para ambas tablas
    excel_bytes_monto = dataframe_to_excel_bytes(monthly_data)
    st.download_button(
        label="Descargar DataFrame en Excel (Proyectado vs Ejecutado por Meses)",
        data=excel_bytes_monto,
        file_name="Proyectado vs Ejecutado por meses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Crear y mostrar el gráfico de líneas con etiquetas
    chart = create_line_chart_with_labels(monthly_data)
    st.altair_chart(chart, use_container_width=True)

    # Calcular la suma acumulada de los datos mensuales
    cumulative_data = calculate_cumulative_sum(monthly_data)
    # También mostrar la suma acumulada
    st.write(f"Suma acumulada de Desembolsos para {year}")
    st.write(cumulative_data)

    cumulative_chart = create_cumulative_line_chart(cumulative_data)
    st.altair_chart(cumulative_chart)

    # Llamar a la función get_monthly_data para obtener los datos mensuales
    monthly_data = get_monthly_data(data, year)

    create_comparison_bar_chart(filtered_data, year)

    create_responsible_comparison_chart(filtered_data, year)

if __name__ == "__main__":
    main()