import streamlit as st
import pandas as pd
import joblib
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predição de Obesidade", 
    layout="wide"
)

# -----------------------------------------------------------------------------
# ESTILIZAÇÃO CSS
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stForm {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://www.fiap.com.br/wp-content/themes/fiap2016/images/logo-fiap.png", 
        width=200
    )
    st.title("Tech Challenge - Fase 4")
    st.subheader("Análise de Obesidade")
    
    st.info("""
    **Objetivo do Projeto:**
    Este sistema utiliza modelos de Machine Learning para auxiliar profissionais de saúde 
    na identificação e classificação de níveis de obesidade com base em hábitos de vida 
    e características físicas.
    """)
    
    st.markdown("---")
    st.markdown("**Integrantes:**")
    st.markdown("- Grupo 33")
    st.markdown("- Pós-Tech Data Analytics")
    
    st.markdown("---")
    st.markdown("**Tecnologias:**")
    st.code("Python\nStreamlit\nScikit-learn\nGoogle Gemini")

# -----------------------------------------------------------------------------
# CARREGAMENTO DO MODELO (CORREÇÃO APLICADA AQUI)
# -----------------------------------------------------------------------------
# O uso de os.path.abspath(__file__) garante que o Python encontre os arquivos
# na pasta correta, não importa onde o terminal foi aberto.
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construção dos caminhos absolutos
model_path = os.path.join(base_dir, 'obesity_prediction_model_pipeline.pkl')
label_path = os.path.join(base_dir, 'label_encoder.pkl')

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_path)
except FileNotFoundError:
    st.error(f"""
        Erro Crítico: Arquivos não encontrados.
        
        O sistema procurou na pasta: 
        {base_dir}
        
        Certifique-se de que os arquivos:
        - obesity_prediction_model_pipeline.pkl
        - label_encoder.pkl
        
        Estão exatamente dentro desta pasta.
    """)
    st.stop()
except Exception as e:
    st.error(f"Erro técnico ao carregar os modelos: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# TÍTULO E INTRODUÇÃO
# -----------------------------------------------------------------------------
st.title("🩺 Sistema Auxiliar de Diagnóstico: Obesidade")
st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para prever o nível de obesidade 
e Inteligência Artificial (Gemini) para sugerir planos de ação personalizados para o paciente.
""")

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA API KEY
# -----------------------------------------------------------------------------
api_key = st.text_input(
    "Insira sua Gemini API Key:", 
    type="password"
)

if api_key:
    genai.configure(api_key=api_key)

# -----------------------------------------------------------------------------
# FORMULÁRIO DE ENTRADA DE DADOS
# -----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.header("Dados do Paciente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox(
            "Gênero", 
            ["Male", "Female"]
        )
        age = st.number_input(
            "Idade", 
            min_value=1, 
            max_value=120, 
            value=25
        )
        height = st.number_input(
            "Altura (m)", 
            min_value=1.0, 
            max_value=2.5, 
            value=1.70, 
            step=0.01
        )
        weight = st.number_input(
            "Peso (kg)", 
            min_value=10.0, 
            max_value=300.0, 
            value=70.0, 
            step=0.1
        )

    with col2:
        family_history = st.selectbox(
            "Histórico Familiar de Sobrepeso?", 
            ["yes", "no"]
        )
        favc = st.selectbox(
            "Consome alimentos calóricos frequentemente? (FAVC)", 
            ["yes", "no"]
        )
        fcvc = st.slider(
            "Frequência de consumo de vegetais (FCVC) [1-3]", 
            1.0, 
            3.0, 
            2.0
        )
        ncp = st.slider(
            "Número de refeições principais (NCP)", 
            1.0, 
            4.0, 
            3.0
        )

    with col3:
        caec = st.selectbox(
            "Consumo de alimentos entre refeições (CAEC)", 
            ["Sometimes", "Frequently", "Always", "no"]
        )
        smoke = st.selectbox(
            "É fumante?", 
            ["yes", "no"]
        )
        ch2o = st.slider(
            "Consumo de água diário em litros (CH2O)", 
            1.0, 
            3.0, 
            2.0
        )
        scc = st.selectbox(
            "Monitora o consumo de calorias? (SCC)", 
            ["yes", "no"]
        )

    st.markdown("---")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        faf = st.slider(
            "Frequência de atividade física (FAF) [0-3]", 
            0.0, 
            3.0, 
            1.0
        )
    with col5:
        tue = st.slider(
            "Tempo de uso de dispositivos tecnológicos (TUE) [0-2]", 
            0.0, 
            2.0, 
            1.0
        )
    with col6:
        calc = st.selectbox(
            "Consumo de álcool (CALC)", 
            ["Sometimes", "no", "Frequently", "Always"]
        )
        mtrans = st.selectbox(
            "Meio de transporte principal (MTRANS)", 
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
        )

    submitted = st.form_submit_button("Realizar Predição e Gerar Plano")

# -----------------------------------------------------------------------------
# PROCESSAMENTO E PREDIÇÃO
# -----------------------------------------------------------------------------
if submitted:
    # Criação do DataFrame garantindo a estrutura correta
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }])

    # Executa a predição no modelo carregado
    prediction_encoded = model.predict(input_data)
    
    # Decodifica o resultado numérico para o label original (ex: Obesity_Type_I)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # -------------------------------------------------------------------------
    # EXIBIÇÃO DOS RESULTADOS
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Resultado da Análise")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(
            label="Classificação Prevista", 
            value=prediction_label
        )
    
    with col_res2:
        if "Obesity" in prediction_label:
            st.warning(
                "O paciente apresenta níveis de obesidade. "
                "Recomenda-se acompanhamento médico especializado."
            )
        elif "Overweight" in prediction_label:
            st.info(
                "O paciente está na faixa de sobrepeso. "
                "Recomenda-se atenção aos hábitos alimentares."
            )
        else:
            st.success(
                "O paciente apresenta uma classificação dentro dos parâmetros normais."
            )

    # -------------------------------------------------------------------------
    # INTEGRAÇÃO COM GEMINI AI
    # -------------------------------------------------------------------------
    if api_key:
        try:
            with st.spinner("Gerando recomendações personalizadas médica via IA..."):
                gemini_model = GenerativeModel("gemini-1.5-flash")
                
                prompt = f"""
                Como médico especialista em nutrologia, forneça um plano de ação resumido 
                para um paciente com o seguinte perfil:
                
                - Classificação de Obesidade: {prediction_label}
                - Idade: {age} anos
                - Peso: {weight} kg
                - Altura: {height} m
                - Frequência de atividade física: {faf} (escala 0-3)
                
                Estruture a resposta em:
                1. Recomendações Dietéticas
                2. Sugestão de Atividades Físicas
                3. Alerta de Saúde
                
                Seja profissional, direto e acolhedor.
                """
                
                response = gemini_model.generate_content(prompt)
                
                st.markdown("---")
                st.markdown("### 🤖 Sugestão da IA (Gemini)")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Erro ao contatar o Gemini: {e}")
    else:
        st.info(
            "Insira sua Gemini API Key na barra lateral ou no campo acima "
            "para receber recomendações personalizadas da IA."
        )

# -----------------------------------------------------------------------------
# RODAPÉ
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Desenvolvido para o Tech Challenge Fase 4 - Pós-Tech Data Analytics - Grupo 33")