import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Argentina Energy AI",
    page_icon="⚡",
    layout="wide"
)

# --- TÍTULO Y ESTILO ---
st.title("⚡ SADI: Predicción de Demanda Energética (Argentina)")
st.markdown("""
Esta aplicación utiliza una **Red Neuronal LSTM** entrenada con datos históricos de CAMMESA
para predecir el consumo eléctrico del Sistema Argentino de Interconexión (SADI).
""")


# --- CARGA DE ARTEFACTOS (CACHÉ) ---
# Usamos @st.cache_resource para cargar el modelo una sola vez y no en cada clic
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('energy_model.h5')
        scaler = joblib.load('scaler.gz')
        return model, scaler
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None


model, scaler = load_artifacts()

# --- SIDEBAR: SIMULACIÓN DE ENTRADA ---
st.sidebar.header("🎛️ Panel de Control")
st.sidebar.subheader("Simular Historial Reciente")

# Simulemos que el usuario ingresa el consumo de las últimas 24 horas
# Por defecto ponemos un patrón típico
default_values = [
    14000, 13500, 13000, 12500, 12200, 12000, 12500, 13500,
    15000, 16500, 17000, 17200, 17500, 17800, 17500, 17000,
    16500, 16000, 16500, 18000, 19500, 20000, 19000, 18000
]

input_data = []
# Creamos un slider para "perturbar" los datos base
factor = st.sidebar.slider("Ajuste de Demanda (Factor)", 0.8, 1.2, 1.0,
                           help="Multiplica la curva base para simular días de mucho calor o frío.")

# Generamos la lista de 24 horas
with st.expander("Ver datos de entrada (24 horas previas)"):
    for i, val in enumerate(default_values):
        val_adjusted = val * factor
        input_data.append(val_adjusted)
        st.text(f"Hora -{24 - i}: {val_adjusted:.0f} MW")

# --- MOTOR DE PREDICCIÓN ---
if st.button("🔮 Predecir Próxima Hora", type="primary"):
    if model is not None:
        # 1. Preprocesar
        input_array = np.array(input_data).reshape(-1, 1)  # Convertir a columna
        input_scaled = scaler.transform(input_array)  # Escalar entre 0 y 1

        # 2. Dar forma para LSTM (1, 24, 1) -> (Muestras, Pasos, Features)
        input_reshaped = input_scaled.reshape(1, 24, 1)

        # 3. Predecir
        prediction_scaled = model.predict(input_reshaped)

        # 4. Invertir escala
        prediction_mw = scaler.inverse_transform(prediction_scaled)[0][0]

        # --- RESULTADOS ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(label="⚡ Demanda Predicha", value=f"{prediction_mw:,.0f} MW", delta="Próxima Hora")

        with col2:
            # Gráfico rápido
            fig, ax = plt.subplots(figsize=(10, 4))
            # Histórico
            ax.plot(range(24), input_data, label='Últimas 24h', color='blue', marker='.')
            # Predicción (Hora 25)
            ax.plot(24, prediction_mw, label='Predicción IA', color='red', marker='o', markersize=10)

            ax.set_title("Contexto y Predicción")
            ax.set_ylabel("MW")
            ax.set_xlabel("Horas (0 = Hace 24h)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    else:
        st.warning("El modelo no está cargado. Revisa los archivos .h5 y .gz")