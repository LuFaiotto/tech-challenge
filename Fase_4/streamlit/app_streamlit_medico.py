import streamlit as st
import pandas as pd
import joblib
import sqlite3
import numpy as np
import os
import google.generativeai as genai
from sklearn.base import BaseEstimator, TransformerMixin

# --- BMICalculator Class
class BMICalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Evitar divisão por zero para Height
        X_copy['Height'] = X_copy['Height'].replace(0, np.nan) # Substituir 0 por NaN
       

        # Calcular BMI
        X_copy['BMI'] = X_copy['Weight'] / (X_copy['Height'] ** 2)
        return X_copy


def generate_gemini_prompt(predicted_obesity_level, user_input_data):


    # Determina o tipo de conselho (perder peso ou manter peso)
    advice_type = "perder peso" # Padrão por ser a maior porcentagem de necessidade
    if predicted_obesity_level in ['Normal_Weight', 'Insufficient_Weight']:
        advice_type = "manter um peso saudável e melhorar hábitos de vida"
    elif predicted_obesity_level in ['Overweight_Level_I', 'Overweight_Level_II']:
        advice_type = "reduzir o peso e adotar hábitos mais saudáveis"
    elif predicted_obesity_level in ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']:
        advice_type = "perder peso de forma saudável e sustentável"

    # Extrair dados relevantes
    name = user_input_data.get('Name', 'Usuário')
    gender = user_input_data.get('Gender', 'Não especificado')
    age = user_input_data.get('Age', 'Não especificado')
    height = user_input_data.get('Height', 'Não especificado')
    weight = user_input_data.get('Weight', 'Não especificado')
    family_history = user_input_data.get('family_history', 'Não especificado')
    favc = user_input_data.get('FAVC', 'Não especificado')
    fcvc = user_input_data.get('FCVC', 'Não especificado')
    ncp = user_input_data.get('NCP', 'Não especificado')
    caec = user_input_data.get('CAEC', 'Não especificado')
    smoke = user_input_data.get('SMOKE', 'Não especificado')
    ch2o = user_input_data.get('CH2O', 'Não especificado')
    scc = user_input_data.get('SCC', 'Não especificado')
    faf = user_input_data.get('FAF', 'Não especificado')
    tue = user_input_data.get('TUE', 'Não especificado')
    calc = user_input_data.get('CALC', 'Não especificado')
    mtrans = user_input_data.get('MTRANS', 'Não especificado')

    prompt = f"""Olá, sou médico e estou cuidando de um paciente com o nível de obesidade {predicted_obesity_level}.

Com base nas informações, gostaria de um conselho para me ajudar o paciente a {advice_type}. Por favor, forneça recomendações práticas e acionáveis, considerando os seguintes dados:

- Gênero: {gender}
- Idade: {age} anos
- Altura: {height} m
- Peso: {weight} kg
- Histórico familiar de obesidade: {family_history}
- Consumo frequente de alimentos calóricos: {favc}
- Frequência de consumo de vegetais nas refeições (1=Raramente, 2=Às vezes, 3=Sempre): {fcvc}
- Número de refeições principais por dia (1=uma, 2=duas, 3=três, 4=quatro ou mais): {ncp}
- Consumo de lanches entre refeições: {caec}
- Fuma: {smoke}
- Consumo de água por dia (1=<1L, 2=1-2L, 3=>2L): {ch2o}
- Monitora ingestão de calorias: {scc}
- Frequência semanal de atividade física (0=Nunca, 1=1-2 dias, 2=2-4 dias, 3=4-5 dias): {faf}
- Tempo diário usando dispositivos eletrônicos (0=0-2h, 1=3-5h, 2=5-8h): {tue}
- Consumo de bebida alcoólica: {calc}
- Meio de transporte habitual: {mtrans}

Por favor, foque em sugestões relacionadas a dieta, exercícios físicos e mudanças no estilo de vida. Seja o mais detalhado e útil possível, faça a avaliação e traga ideias de médicos e nutricionistas.
"""
    return prompt

# --- carrega modelo ---
try:
    loaded_model_pipeline = joblib.load('obesity_prediction_model_pipeline.pkl')
    loaded_label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Erro: Arquivos do modelo ou do LabelEncoder não encontrados. Certifique-se de que 'obesity_prediction_model_pipeline.pkl' e 'label_encoder.pkl' estão no mesmo diretório do seu script Streamlit.")
    st.stop()
except AttributeError as e:
    st.error(f"Erro de compatibilidade da versão do scikit-learn ao carregar o modelo: {e}. Isso geralmente ocorre se o modelo foi salvo com uma versão diferente do scikit-learn. Por favor, re-execute as células que treinam e salvam o modelo (`FPCIueUFX41Y` a `vgHC6GJAaiEH`) e o label encoder neste ambiente para gerar arquivos compatíveis.")
    st.stop()

# --- Configurações do Gemini API ---

if "GOOGLE_API_KEY" not in st.secrets:
    st.warning("A chave GOOGLE_API_KEY não foi encontrada em `st.secrets`. Por favor, adicione sua chave Gemini API ao seu arquivo `.streamlit/secrets.toml` para usar os conselhos da IA.")
    st.info("Exemplo de `secrets.toml`:\n`GOOGLE_API_KEY = \"SUA_CHAVE_AQUI\"`")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    try:
        gemini_client = genai.GenerativeModel('gemini-2.5-flash') # Modelo otimizado para velocidade e custo
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo Gemini: {e}. Verifique sua GOOGLE_API_KEY em `secrets.toml`.")
        st.stop()

# --- Configuração do Streamlit App  ---
st.set_page_config(
    page_title="Predição de Nível de Obesidade e Conselhos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customização CSS para deixar alguns textos em roxo, cor de prevenção a obesidade.
css = """
<style>
body {
    background-color: #E6E6FA; /* Light purple */
}

h1, h2, h3, h4, h5, h6 {
    color: #4B0082; /* Dark purple */
}

.stMarkdown strong {
    color: #4B0082; /* Dark purple for strong/bold text */
}

.stButton>button {
    background-color: #8A2BE2; /* Bluish purple for buttons */
    color: white;
}
.stButton>button:hover {
    background-color: #9370DB; /* Lighter purple on hover */
    color: white;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title("Avaliação do Nível de Obesidade e Conselhos Personalizados")
st.write("Preencha as informações do paciente abaixo para obter sua predição e recomendações.")

# --- dados dos pacientes (Área principal) ---
user_input_data = {}

user_input_data['Name'] = st.text_input("Qual o nome do paciente?", "Paciente")

gender_map = {
    'Masculino': 'Male',
    'Feminino': 'Female'
}
gender_display_options = list(gender_map.keys())
gender_selected_pt = st.radio("Qual o gênero?", gender_display_options)
user_input_data['Gender'] = gender_map[gender_selected_pt]

user_input_data['Age'] = st.number_input("Qual a idade? (14-61)", min_value=14, max_value=61, value=30, step=1)
user_input_data['Height'] = st.number_input("Qual a altura? (m, ex: 1.75)", min_value=1.0, max_value=2.5, value=1.70, format="%.2f")
user_input_data['Weight'] = st.number_input("Qual é o peso? (kg, ex: 70.5)", min_value=30.0, max_value=300.0, value=70.0, format="%.1f")

yes_no_map = {
    'Sim': 'yes',
    'Não': 'no'
}
family_history_display_options = list(yes_no_map.keys())
family_history_selected_pt = st.radio("Tem histórico de obesidade na família?", family_history_display_options)
user_input_data['family_history'] = yes_no_map[family_history_selected_pt]

favc_selected_pt = st.radio("Consumo frequente de alimentos muito calóricos?", family_history_display_options, key="favc_radio") # Reusa opções
user_input_data['FAVC'] = yes_no_map[favc_selected_pt]

user_input_data['FCVC'] = st.selectbox("Frequência de consumo de vegetais nas refeições?", [1.0, 2.0, 3.0], format_func=lambda x: {1.0: "1: Raramente", 2.0: "2: Às vezes", 3.0: "3: Sempre"}[x], key="fcvc_select")

user_input_data['NCP'] = st.selectbox("Qual o número de refeições principais por dia?", [1.0, 2.0, 3.0, 4.0], format_func=lambda x: {1.0: "1: uma refeição", 2.0: "2: duas", 3.0: "3: três", 4.0: "4: quatro ou mais"}[x], key="ncp_select")

caec_calc_map = {
    'Não': 'no',
    'Às vezes': 'Sometimes',
    'Frequentemente': 'Frequently',
    'Sempre': 'Always'
}
caec_display_options = list(caec_calc_map.keys())
caec_selected_pt = st.selectbox("Consome lanches/comes entre as refeições?", caec_display_options, key="caec_select")
user_input_data['CAEC'] = caec_calc_map[caec_selected_pt]

smoke_selected_pt = st.radio("Fuma?", family_history_display_options, key="smoke_radio") # Reusa opções
user_input_data['SMOKE'] = yes_no_map[smoke_selected_pt]

user_input_data['CH2O'] = st.selectbox("O quanto de água consome por dia?", [1.0, 2.0, 3.0], format_func=lambda x: {1.0: "1: <1L", 2.0: "2: 1-2L", 3.0: ">2L"}[x], key="ch2o_select")

scc_selected_pt = st.radio("Monitora sua ingestão de calorias diárias?", family_history_display_options, key="scc_radio") # Reusa opções
user_input_data['SCC'] = yes_no_map[scc_selected_pt]

user_input_data['FAF'] = st.selectbox("Qual a frequência semanal de atividade física?", [0.0, 1.0, 2.0, 3.0], format_func=lambda x: {0.0: "0: Nunca", 1.0: "1: 1-2 dias", 2.0: "2: 2-4 dias", 3.0: "3: 4-5 dias"}[x], key="faf_select")

user_input_data['TUE'] = st.selectbox("Qual o tempo diário usando dispositivos eletrônicos?", [0.0, 1.0, 2.0], format_func=lambda x: {0.0: "0: 0-2 horas", 1.0: "1: 3-5 horas", 2.0: "2: 5-8 horas"}[x], key="tue_select")

calc_display_options = list(caec_calc_map.keys())
calc_selected_pt = st.selectbox("Consome bebida alcoólica?", calc_display_options, key="calc_select")
user_input_data['CALC'] = caec_calc_map[calc_selected_pt]

mtrans_map = {
    'Transporte Público': 'Public_Transportation',
    'Andar': 'Walking',
    'Automóvel': 'Automobile',
    'Outros': 'Other',
    'Bicicleta': 'Bike'
}
mtrans_display_options = list(mtrans_map.keys())
mtrans_selected_pt = st.selectbox("Qual o meio de transporte habitual?", mtrans_display_options, key="mtrans_select")
user_input_data['MTRANS'] = mtrans_map[mtrans_selected_pt]

predict_button = st.button("Obter Predição e Conselhos")

# --- Área principal com resultados ---
if predict_button:
    st.subheader("Resultados:")

    
    user_df = pd.DataFrame([user_input_data])
    
    user_df_for_model = user_df.drop(columns=['Name'])

    try:
        
        predicted_label_encoded = loaded_model_pipeline.predict(user_df_for_model)
        predicted_obesity_level = loaded_label_encoder.inverse_transform(predicted_label_encoded)[0]

        st.success(f"Para o paciente {user_input_data['Name']}, o nível de obesidade previsto é: **{predicted_obesity_level.replace('_', ' ')}**")

        
        gemini_prompt = generate_gemini_prompt(predicted_obesity_level, user_input_data)

        if "GOOGLE_API_KEY" in st.secrets:
            with st.spinner("Gerando conselhos personalizados..."):
                gemini_response_obj = gemini_client.generate_content(gemini_prompt)
                gemini_response_text = gemini_response_obj.text

            st.subheader("Conselhos Personalizados:")
            st.markdown(gemini_response_text)
        else:
            st.warning("Não foi possível gerar conselhos da IA Gemini. GOOGLE_API_KEY não configurada.")
            gemini_response_text = "Conselhos da IA Gemini não disponíveis." # Default text if API not configured

        # --- Salvar no banco de dados a resposta ---
        sqlite_file = 'obesity_data.db'
        conn = sqlite3.connect(sqlite_file)

        
        user_df['Results'] = predicted_obesity_level

        try:
            max_id_query_user = f"SELECT MAX(ID_R) FROM respostas"
            last_id_user = pd.read_sql_query(max_id_query_user, conn).iloc[0, 0]
            next_id_user = int(last_id_user) + 1 if pd.notnull(last_id_user) else 1
        except Exception: 
            next_id_user = 1

        
        user_df.insert(0, 'ID_R', next_id_user)
        user_df.to_sql('respostas', conn, if_exists='append', index=False)
        st.info(f"As respostas do paciente e a predição foram salvas no banco de dados com ID_R: {next_id_user}.")

        
        gemini_table_name = 'gemini_responses'
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {gemini_table_name} (
                ID_G INTEGER PRIMARY KEY AUTOINCREMENT,
                Gemini_Response TEXT
            );
        ''')

        try:
            max_id_query_gemini = f"SELECT MAX(ID_G) FROM {gemini_table_name}"
            last_id_gemini = pd.read_sql_query(max_id_query_gemini, conn).iloc[0, 0]
            next_id_gemini = int(last_id_gemini) + 1 if pd.notnull(last_id_gemini) else 1
        except Exception:
            next_id_gemini = 1

        insert_gemini_query = f"INSERT INTO {gemini_table_name} (ID_G, Gemini_Response) VALUES (?, ?)"
        conn.execute(insert_gemini_query, (next_id_gemini, gemini_response_text))

        conn.commit()
        conn.close()
        st.info(f"O conselho foi salvo no banco de dados para acompanhamento médico.")

    except Exception as e:
        st.error(f"Ocorreu um erro durante a predição ou geração de conselhos: {e}")

else:
    st.info("Clique no botão 'Obter Predição e Conselhos' para processar suas informações!")
