"""
Checkpoint 4 — App Streamlit para Previsão de Aluguel
Carrega o pipeline treinado (modelo/modelo.pkl) e permite previsões interativas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- Configuração da página ---
st.set_page_config(
    page_title="Previsão de Aluguel — CP4",
    page_icon="🏠",
    layout="centered",
)

# --- CSS: limpo, profissional, sem exagero ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    h1 {
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        color: #1a1a1a;
    }

    .result-box {
        background-color: #f7f7f7;
        border-left: 4px solid #1174E6;
        padding: 16px 20px;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 12px;
        color: #1a1a1a;
    }

    .alert-box {
        background-color: #fffbe6;
        border-left: 4px solid #e6a800;
        padding: 10px 14px;
        font-size: 0.85rem;
        margin-bottom: 12px;
        color: #5c4800;
    }
</style>
""", unsafe_allow_html=True)

# --- Carregar modelo e metadados ---
MODELO_PATH = "modelo/modelo.pkl"
METADADOS_PATH = "modelo/metadados.json"


@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODELO_PATH):
        st.error(f"Arquivo {MODELO_PATH} não encontrado. Rode o notebook primeiro.")
        st.stop()
    return joblib.load(MODELO_PATH)


@st.cache_data
def carregar_metadados():
    if not os.path.exists(METADADOS_PATH):
        st.error(f"Arquivo {METADADOS_PATH} não encontrado. Rode o notebook primeiro.")
        st.stop()
    with open(METADADOS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


pipeline = carregar_modelo()
metadados = carregar_metadados()

ranges = metadados["ranges_treino"]
cidades = metadados["cidades_validas"]

# --- Header ---
st.markdown("# Previsão de Aluguel de Imóveis")
st.caption(
    f"Modelo: {metadados['modelo_nome']} · "
    f"R² = {metadados['metricas_teste']['R2']:.4f} · "
    f"MAE = R$ {metadados['metricas_teste']['MAE']:,.2f}"
)

st.markdown("---")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("Cidade", options=cidades)
    area = st.number_input("Área (m²)", min_value=10, max_value=5000, value=80, step=10)
    rooms = st.number_input("Quartos", min_value=1, max_value=10, value=2, step=1)
    bathroom = st.number_input("Banheiros", min_value=1, max_value=8, value=1, step=1)
    parking = st.number_input("Vagas de garagem", min_value=0, max_value=8, value=1, step=1)

with col2:
    floor = st.number_input("Andar", min_value=0, max_value=50, value=0, step=1)
    animal = st.selectbox("Aceita animais", options=["Não", "Sim"])
    furniture = st.selectbox("Mobiliado", options=["Não", "Sim"])
    hoa = st.number_input("Condomínio (R$)", min_value=0, max_value=15000, value=500, step=50)
    populacao_estimada = st.number_input(
        "População da cidade",
        min_value=100000, max_value=15000000, value=5000000, step=100000,
    )

st.markdown("")

# --- Predição ---
if st.button("Calcular previsão", use_container_width=True):
    animal_val = 1 if animal == "Sim" else 0
    furniture_val = 1 if furniture == "Sim" else 0

    input_data = pd.DataFrame([{
        "city": city,
        "area": area,
        "rooms": rooms,
        "bathroom": bathroom,
        "parking spaces": parking,
        "floor": floor,
        "animal": animal_val,
        "furniture": furniture_val,
        "hoa (R$)": hoa,
        "populacao_estimada": populacao_estimada,
    }])

    # Alerta de extrapolação
    alertas = []
    for col, limites in ranges.items():
        valor = input_data[col].iloc[0]
        if valor < limites["min"] or valor > limites["max"]:
            alertas.append(
                f"<b>{col}</b>: {valor} (treino: {limites['min']:.0f} – {limites['max']:.0f})"
            )

    if alertas:
        alerta_html = "<div class='alert-box'>Valores fora do range de treino:<br>"
        alerta_html += "<br>".join(alertas)
        alerta_html += "</div>"
        st.markdown(alerta_html, unsafe_allow_html=True)

    # Resultado
    predicao = pipeline.predict(input_data)[0]
    st.markdown(
        f"<div class='result-box'>Aluguel estimado: R$ {predicao:,.2f}</div>",
        unsafe_allow_html=True,
    )

    st.caption("Estimativa baseada em dados históricos. O valor real pode variar.")
