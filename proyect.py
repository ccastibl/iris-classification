import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

# -----------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])

# Codificar especies
le = LabelEncoder()
df["Species_cod"] = le.fit_transform(df["Species"])

# Variables independientes y dependiente
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species_cod']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------
# FUNCIONES
# -----------------------------------------------------
def calcular_metricas():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    return accuracy, precision, recall, f1

def predecir_especie(values):
    pred = model.predict([values])[0]
    return le.inverse_transform([pred])[0]

# -----------------------------------------------------
# DASHBOARD STREAMLIT
# -----------------------------------------------------
st.title("🌸 Iris Species Classification Dashboard")
st.write("Proyecto de clasificación utilizando el dataset de Iris.")

menu = st.sidebar.radio("Navegación", ["📊 Métricas del Modelo", "🌿 Visualizaciones", "🔮 Predicción del Usuario"])

# -----------------------------------------------------
# 📊 MÉTRICAS DEL MODELO
# -----------------------------------------------------
if menu == "📊 Métricas del Modelo":
    st.header("📊 Métricas del Modelo")

    accuracy, precision, recall, f1 = calcular_metricas()

    st.write("### Resultados:")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

# -----------------------------------------------------
# 🌿 VISUALIZACIONES
# -----------------------------------------------------
elif menu == "🌿 Visualizaciones":
    st.header("🌿 Visualización de Datos")

    numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    st.write("### Histogramas")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=20, title=f"Histograma de {col}")
        st.plotly_chart(fig)

    st.write("### Gráfico 3D del dataset")
    fig3d = px.scatter_3d(
        df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        z="SepalLengthCm",
        color="Species",
        title="Dataset Iris - Visualización 3D"
    )
    st.plotly_chart(fig3d)

# -----------------------------------------------------
# 🔮 PREDICCIÓN DEL USUARIO
# -----------------------------------------------------
elif menu == "🔮 Predicción del Usuario":
    st.header("🔮 Predicción de Especie")

    sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

    if st.button("Predecir especie"):
        values = [sl, sw, pl, pw]
        pred = predecir_especie(values)
        st.success(f"La especie predicha es: **{pred}**")

        # 3D con punto del usuario
        df_aux = df.copy()
        df_aux["Tipo"] = "Dataset"
        new_point = pd.DataFrame({
            "PetalLengthCm": [pl],
            "PetalWidthCm": [pw],
            "SepalLengthCm": [sl],
            "Species": [pred],
            "Tipo": ["Nuevo Punto"]
        })
        df_plot = pd.concat([df_aux, new_point])

        fig_pred = px.scatter_3d(
            df_plot,
            x="PetalLengthCm",
            y="PetalWidthCm",
            z="SepalLengthCm",
            color="Tipo",
            symbol="Tipo",
            title="Posición del Punto Predicho"
        )
        st.plotly_chart(fig_pred)
