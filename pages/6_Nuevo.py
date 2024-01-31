import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Función para cargar datos
@st.cache
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    return data

# Función para preprocesar datos
def preprocess_data(data):
    X = data[['Año', 'País', 'AreaPrioritaria', 'AreaIntervencion']]
    Y = data['PorcentajeAcumulado']
    categorical_features = ['País', 'AreaPrioritaria', 'AreaIntervencion']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    X_processed = preprocessor.fit_transform(X)
    return train_test_split(X_processed, Y, test_size=0.2, random_state=0)

# Función para entrenar y evaluar modelos
def train_and_evaluate(X_train, X_test, Y_train, Y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Random Forest": RandomForestRegressor(random_state=0),
        "Support Vector Machine": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        results[name] = {"MSE": mse, "R^2": r2}
    return results

def main():
# Inicio de la aplicación Streamlit
    st.title('Análisis y Modelado de Datos')

    # Carga de datos
    uploaded_file = st.file_uploader("Carga tu archivo Excel aquí", type=["xlsx"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write(data.head())

        # Preprocesamiento de datos
        X_train, X_test, Y_train, Y_test = preprocess_data(data)

        # Entrenamiento y evaluación de modelos
        if st.button('Entrenar Modelos'):
            results = train_and_evaluate(X_train, X_test, Y_train, Y_test)
            st.write(results)

if __name__ == "__main__":
    main()